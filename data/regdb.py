import os
import os.path as osp
import glob
import re
import random


class RegDB(object):
    """
    RegDB
    Reference:

    Dataset statistics:
    # identities:
    # images:
    """

    def __init__(self, root='/home/lyq/Desktop/dataset/Reid/RegDB', trial='1', verbose=True, **kwargs):
        random.seed(2019)

        super(RegDB, self).__init__()
        self.dataset_dir = root
        self.trial = trial
        train_RGB, train_IR  = self._process_dir('train', root)
        gallery, query  = self._process_dir('test', root)

        if verbose:
            print("=> RegDB loaded")
            self.print_dataset_statistics(train_RGB, train_IR, query, gallery)

        self.train_RGB  = train_RGB
        self.train_IR  = train_IR
        self.query = query
        self.gallery = gallery

        self.num_train_RGB_pids, self.num_train_RGB_imgs = self.get_imagedata_info_r(self.train_RGB)
        self.num_train_IR_pids, self.num_train_IR_imgs   = self.get_imagedata_info_r(self.train_IR)

        self.num_query_pids, self.num_query_imgs  = self.get_imagedata_info_r(self.query)
        self.num_gallery_pids, self.num_gallery_imgs = self.get_imagedata_info_r(self.gallery)

    def _process_dir(self, stage, root, relabel=False):
        if stage == 'train':
            RGB_file_list = root + '/idx/train_visible_{}'.format(self.trial) + '.txt'
            IR_file_list  = root + '/idx/train_thermal_{}'.format(self.trial) + '.txt'
        elif stage == 'test':
            RGB_file_list = root + '/idx/test_visible_{}'.format(self.trial) + '.txt'
            IR_file_list = root + '/idx/test_thermal_{}'.format(self.trial) + '.txt'
        else:
            raise RuntimeError("'{}' is not available".format(stage))

        pid_container = set()
        RGB_dataset, IR_dataset = [], []
        with open(RGB_file_list) as f:
            data_file_list = f.read().splitlines()
            for s in data_file_list:
                RGB_dataset.append((root+'/'+s.split(' ')[0], int(s.split(' ')[1]), 2))
        with open(IR_file_list) as f:
            data_file_list = f.read().splitlines()
            for s in data_file_list:
                IR_dataset.append((root + '/' + s.split(' ')[0], int(s.split(' ')[1]), 1))

        return RGB_dataset, IR_dataset

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def get_imagedata_info_r(self, data):
        pids = []
        for _, pid, _ in data:
            pids += [pid]
        pids = set(pids)
        num_pids = len(pids)
        num_imgs = len(data)
        return num_pids, num_imgs

    def print_dataset_statistics(self, train_RGB,train_IR, query, gallery):
        num_train_RGB_pids, num_train_RGB_imgs = self.get_imagedata_info_r(train_RGB)
        num_train_IR_pids, num_train_IR_imgs   = self.get_imagedata_info_r(train_IR)
        num_query_pids, num_query_imgs         = self.get_imagedata_info_r(query)
        num_gallery_pids, num_gallery_imgs     = self.get_imagedata_info_r(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images ")
        print("  ----------------------------------------")
        print("  train RGB| {:5d} | {:8d} ".format(num_train_RGB_pids, num_train_RGB_imgs ))
        print("  train IR | {:5d} | {:8d} ".format(num_train_IR_pids, num_train_IR_imgs ))
        print("  query    | {:5d} | {:8d} ".format(num_query_pids, num_query_imgs ))
        print("  gallery  | {:5d} | {:8d} ".format(num_gallery_pids, num_gallery_imgs ))
        print("  ----------------------------------------")