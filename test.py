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
from utils.eval_reid import *

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
    parser.add_argument("--config", default="./configs/res50_regdb_all.ini", help="path to config file", type=str)
    parser.add_argument("--model_path", default="./output/res50_sysumm/all_best.pth", help="path to test model", type=str)
    parser.add_argument("--trial", default="1", help="which trial to test", type=str)
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
    train_loader, val_loader, num_query, num_classes = make_dataloader(config, config['DATA']['DATASET_NAME'], trial = parser.parse_args().trial)
    model = build_model(config, num_classes).to(device)

    # ------------init logger-------------#
    logger = setup_logger(config['MODEL']['LOGGER_NAME'], config['MODEL']['OUTPUT_PATH'],
                          config['MODEL']['LOGGER_NAME']+".txt",0)
    logger = logging.getLogger(config['MODEL']['LOGGER_NAME'])
    logger.info("Init Done")

    # --------------training--------------#
    model = torch.load(parser.parse_args().model_path).to(device)
    model.eval()
    rank_list = [0, 0, 0, 0]
    eval_times = len(val_loader)
    for i in range(eval_times):
        feats, pids, camids, img_paths =[], [], [], []
        for data, pid, camid, img_path in val_loader[i]:
            with torch.no_grad():
                data = data.to(device)
                feat, _ = model(data)
            feats.append(feat)
            pids.extend(pid)
            camids.extend(camid)
            img_paths.extend(img_path)

        rank_score, rank_map = R1_mAP(num_query, feats, pids, camids, dataset=config['DATA']['DATASET_NAME'])
        result = [rank_score[0], rank_score[9], rank_score[19], rank_map]
        rank_list = [rank_list[i]+result[i] for i in range(4)]
        logger.info("val iter:{} val rank1: {:.5f}, rank10: {:.5f}, rank20: {:.5f} val map: {:.5f}"
                    .format(i, result[0], result[1], result[2], result[3]))
    rank_list = [i / eval_times for i in rank_list]
    logger.info("size: {} query size: {} gallery size: {} \n\
                 val rank1: {:.5f}, rank10: {:.5f}, rank20: {:.5f} val map: {:.5f}"
                .format(len(val_loader[i].dataset), num_query, len(val_loader[i].dataset) - num_query,
                        rank_list[0], rank_list[1], rank_list[2], rank_list[3]))
