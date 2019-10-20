import argparse
import os
import sys
import torch
import configparser
import logging
import numpy as np
sys.path.append('.')
import random
from torch.backends import cudnn
from torch import nn
from data.data_loader import make_dataloader
from model import *
from slover import make_loss, make_optimizer, WarmupMultiStepLR
from utils.logger import setup_logger
from utils.eval_reid import R1_mAP
from torch.nn import functional as F

def random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    # -------prepare config option---------#
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/res50_sysumm_all.ini", help="path to config file", type=str)
    config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
    config.read(parser.parse_args().config)
    # -------init environment-------------#
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    output_path = config['MODEL']['OUTPUT_PATH']
    if output_path and not os.path.exists(output_path): os.makedirs(output_path)
    if config['MODEL']['DEVICE'] == "cuda": os.environ['CUDA_VISIBLE_DEVICES'] = config['MODEL']['DEVICE_ID']
    cudnn.benchmark = True
    device = config['MODEL']['DEVICE']
    print("Using DEVICE {} : {}".format(device, config['MODEL']['DEVICE_ID']))
    # -------fix random seed-------------#
    seed = int(config['MODEL']['SEED'])
    random_seed(seed)
    # --------prepare dataset model-------#
    train_loader, val_loader, num_query, num_classes = make_dataloader(config, config['DATA']['DATASET_NAME'])
    model = build_model(config, num_classes).to(device)
    ide_net = bulid_classifier(num_classes).to(device)

    # ---prepare loss optimizer lr_scheduler---#
    optimizer = make_optimizer(config, model)
    optimizer_id = make_optimizer(config, ide_net, 'head')
    id_loss, triplet_loss = make_loss(config, num_classes)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    # ------------init logger-------------#
    logger = setup_logger(config['MODEL']['LOGGER_NAME'], config['MODEL']['OUTPUT_PATH'],
                          config['MODEL']['LOGGER_NAME']+".txt",0)
    logger = logging.getLogger(config['MODEL']['LOGGER_NAME'])
    logger.info("Init Done")
    # --------get modal index-------------#
    num_instance = int(config['DATA']['SAMPLE_ID_INSTANCE'])
    num_batch_size = int(config['DATA']['TRAIN_BATCH_SIZE'])
    rgb, ir = [], []
    for i in range(0, num_batch_size, 2 * num_instance):
        ir.extend([a for a in range(i, i + num_instance)])
        rgb.extend([b for b in range(i + num_instance, i + 2 * num_instance)])
    rgb, ir = torch.LongTensor(rgb).to(device), torch.LongTensor(ir).to(device)

    # --------------training--------------#
    train_loss, train_acc = [], []
    num_epochs = int(config['TRAIN']['MAX_EPOCHS'])
    best_result = [0,0,0,0]

    for epoch in range(num_epochs):

        model.train()
        ide_net.train()
        train_epoch_id_loss, trian_epoch_kl_loss, train_epoch_metric_loss, train_epoch_correct = 0, 0, 0, 0
        iter_size = 0
        for iter, (data, pids, _, img_path) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_id.zero_grad()

            data, pids = data.to(device), pids.to(device)
            score, feat = model(data)
            ide_score = ide_net(feat, 'fix')
            rgb_score, ir_score = torch.index_select(ide_score[0], 0, rgb), torch.index_select(ide_score[2], 0, ir)
            rgb_tgt, ir_tgt = torch.index_select(ide_score[2], 0, rgb).detach(), torch.index_select(ide_score[0], 0, ir).detach()
            rgb_pid, ir_pid = torch.index_select(pids, 0, rgb), torch.index_select(pids, 0, ir)
            loss_kl = kl_loss(F.log_softmax(rgb_score, dim=1), F.softmax(rgb_tgt, dim=1)) + kl_loss(F.log_softmax(ir_score, dim=1), F.softmax(ir_tgt, dim=1))
            loss_id = id_loss(score, feat, pids) + id_loss(rgb_score, feat, rgb_pid) + id_loss(ir_score, feat, ir_pid)
            trian_epoch_kl_loss += loss_kl.item()
            train_epoch_id_loss += loss_id.item()
            loss_metric = triplet_loss(feat, pids)
            loss = loss_kl + loss_id + loss_metric
            train_epoch_metric_loss += loss_metric.item()

            loss.backward()
            optimizer_id.step()
            optimizer.step()
            correct = (score.max(1)[1] == pids).sum()
            train_epoch_correct += correct.item()

            iter_size = iter

        data_size = train_loader.batch_size * (iter_size + 1)
        if train_epoch_id_loss != 0: train_epoch_id_loss = train_epoch_id_loss / data_size
        if trian_epoch_kl_loss != 0: trian_epoch_kl_loss = trian_epoch_kl_loss / data_size
        if train_epoch_metric_loss != 0: train_epoch_metric_loss = train_epoch_metric_loss / data_size
        train_epoch_acc = train_epoch_correct / data_size

        logger.info("Epoch[{}] train size: {} train Acc: {:.3f} ID Loss: {:.5f}, KL Loss: {:.5f} Metric Loss: {:.5f} "
                    .format(epoch, data_size, train_epoch_acc, train_epoch_id_loss,trian_epoch_kl_loss, train_epoch_metric_loss))

        # -------evaluate----------#
        if (epoch + 1) % int(config['TRAIN']['EVAL_PERIOD']) == 0:
            model.eval()
            ide_net.eval()
            rank_list = [0, 0, 0, 0]
            eval_times = len(val_loader)
            for i in range(eval_times):
                feats, pids, camids = [], [], []
                for data, pid, camid, _ in val_loader[i]:
                    with torch.no_grad():
                        data = data.to(device)
                        feat, _ = model(data)
                    feats.append(feat)
                    pids.extend(pid)
                    camids.extend(camid)

                rank_score, rank_map = R1_mAP(num_query, feats, pids, camids, dataset=config['DATA']['DATASET_NAME'])
                result = [rank_score[0], rank_score[9], rank_score[19], rank_map]
                rank_list = [rank_list[i]+result[i] for i in range(4)]
                logger.info("Epoch[{}] val iter:{} val rank1: {:.5f}, rank10: {:.5f}, rank20: {:.5f} val map: {:.5f}"
                            .format(epoch, i, result[0], result[1], result[2], result[3]))
            rank_list = [i / eval_times for i in rank_list]
            logger.info("Epoch[{}] size: {} query size: {} gallery size: {} \n\
                         val rank1: {:.5f}, rank10: {:.5f}, rank20: {:.5f} val map: {:.5f}"
                        .format(epoch, len(val_loader[i].dataset), num_query, len(val_loader[i].dataset) - num_query,
                                rank_list[0], rank_list[1], rank_list[2], rank_list[3]))

            if rank_list[0] > best_result[0]:
                best_result = rank_list
                # -----save best model------#
                save_filename = '{}_best.pth'.format(config['MODEL']['LOGGER_NAME'])
                save_path = os.path.join(config['MODEL']['OUTPUT_PATH'], save_filename)
                torch.save(model, save_path)
        # # -----save checkpoint------#
        # if (epoch + 1) % int(config['TRAIN']['CHECKPOINT_PERIOD']) == 0:
        #     save_filename = '{}_{}.pth'.format(config['MODEL']['LOGGER_NAME'], epoch)
        #     save_path = os.path.join(config['MODEL']['OUTPUT_PATH'], save_filename)
        #     torch.save(model, save_path)

    logger.info("Max val rank1: {:.5f}, rank10: {:.5f}, rank20: {:.5f} val map: {:.5f} "
                .format(best_result[0], best_result[1], best_result[2], best_result[3] ))
