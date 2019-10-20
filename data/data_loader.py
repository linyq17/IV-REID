import os.path as osp
import configparser
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .datasets import *
from .sampler import *
from .sysumm  import SYSUMM
from .regdb   import RegDB
from .transforms import build_transforms
# from sampler import *
# from sysumm  import SYSUMM
# from regdb   import RegDB
# from transforms import build_transforms
# from datasets import *

def org_train_collate_fn(batch):
    imgs, pids, camid, img_path, = [], [], [], []
    for data in batch:
        imgs.append(data[0]['real'])
        pids.append(data[1])
        camid.append(data[2])
        img_path.append(data[3]['real'])

    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camid, img_path

def gen_train_collate_fn(batch):
    imgs, pids, camid, img_path, = [], [], [], []
    for data in batch:
        if len(data[0]) != 1:
            imgs.append(data[0]['real'])
            imgs.append(data[0]['fake'])
            pids.extend([data[1],data[1]])
            camid.extend([data[2],data[2]])
            img_path.append(data[3]['real'])
            img_path.append(data[3]['fake'])
        else:
            imgs.append(data[0]['real'])
            pids.append(data[1])
            camid.append(data[2])
            img_path.append(data[3]['real'])

    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camid, img_path

def grey_train_collate_fn(batch):
    imgs, pids, camid, img_path, = [], [], [], []
    for data in batch:
        if len(data[0]) != 1:
            imgs.append(data[0]['grey'])
            pids.append(data[1])
            camid.append(data[2])
            img_path.append(data[3]['grey'])
        else:
            imgs.append(data[0]['real'])
            pids.append(data[1])
            camid.append(data[2])
            img_path.append(data[3]['real'])

    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camid, img_path

def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths

def make_dataloader(config, dataset_name='SYSUMM', trial=1):
    seed = int(config['MODEL']['SEED'])

    train_transforms = build_transforms(config, is_train=True)
    val_transforms   = build_transforms(config, is_train=False)

    if dataset_name == 'SYSUMM':
        dataset = SYSUMM(seed=seed)
        train_set = SYSU_ImageDataset(dataset.train_RGB + dataset.train_IR, train_transforms)

    elif dataset_name == 'RegDB':
        dataset = RegDB(trial=trial)
        train_set = RegDB_ImageDataset(dataset.train_RGB + dataset.train_IR, train_transforms)

    num_classes = dataset.num_train_RGB_pids
    SAMPLER = config['DATA']['SAMPLER']

    if SAMPLER == 'random':
        print('Using Random INPUT')
        train_loader = DataLoader(
            train_set,
            batch_size=int(config['DATA']['TRAIN_BATCH_SIZE']),
            shuffle=True,
            num_workers=int(config['DATA']['NUM_WORKERS']),
            collate_fn=org_train_collate_fn
        )
    elif SAMPLER == 'balanced':
        print('Using Balanced INPUT with Original images')
        train_loader = DataLoader(
            train_set,
            batch_size=int(config['DATA']['TRAIN_BATCH_SIZE']),
            sampler=BalancedSampler(data_source=dataset.train_RGB + dataset.train_IR,
                                    batch_size=int(config['DATA']['TRAIN_BATCH_SIZE']),
                                    num_instances=int(config['DATA']['SAMPLE_ID_INSTANCE']),
                                    dataset=dataset_name,
                                    seed=seed),
            num_workers=int(config['DATA']['NUM_WORKERS']),
            collate_fn=org_train_collate_fn
        )
    else:
        raise KeyError("Unknown SAMPLER: {}".format(SAMPLER))
    print("training data iters: {} with sampler:{}".format(len(train_loader), SAMPLER))

    val_loader = []
    if dataset_name == 'SYSUMM':
        for i in range(len(dataset.gallery)):
            val_set = ImageDataset(dataset.query + dataset.gallery[i], val_transforms)
            val_loader.append(DataLoader(
                val_set,
                batch_size=int(config['DATA']['TEST_BATCH_SIZE']),
                shuffle=False,
                num_workers=int(config['DATA']['NUM_WORKERS']),
                collate_fn=val_collate_fn
            ))

    elif dataset_name == 'RegDB':
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        val_loader.append(DataLoader(
            val_set,
            batch_size=int(config['DATA']['TEST_BATCH_SIZE']),
            shuffle=False,
            num_workers=int(config['DATA']['NUM_WORKERS']),
            collate_fn=val_collate_fn
        ))

    return train_loader, val_loader, len(dataset.query), num_classes

# if __name__ == '__main__':
#     config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
#     config.read('../configs/res50_sysumm/grey_id.ini')
#
#     print(list(map(int,config['DATA'].getlist('SIZE_TRAIN'))))
#     train_loader, val_loader, _, num_classes = make_dataloader(config,dataset_name='SYSUMM')
#     print(num_classes)
#     for iter, (_, pids, camid, img_path) in enumerate(val_loader):
#         if iter > 60:
#             print(iter)
#             print(pids)
#             print(camid)
#             print(img_path)
#
#             break