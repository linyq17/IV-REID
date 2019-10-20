import os
import os.path as osp
import glob
import re
import random

class SYSUMM(object):
    """
    SYSU-MM
    Reference:

    Dataset statistics:
    # identities:
    # images:
    """

    def __init__(self, root='/home/lyq/Desktop/dataset/Reid/SYSU-MM', verbose=True, seed=2019, **kwargs):
    # def __init__(self, root='/data/Reid/SYSU-MM', verbose=True, **kwargs):

        super(SYSUMM, self).__init__()
        self.dataset_dir = root
        self.seed = seed
        random.seed(seed)

        train_RGB  = self._process_dir('train', 'RGB', root, relabel=True)
        train_IR  = self._process_dir('train', 'IR', root, relabel=True)
        query   = self._process_dir('test', 'IR', root)
        gallery = [self._process_dir('test','RGB', root,seed=seed+i*33) for i in range(10)]

        random.seed(seed)
        if verbose:
            print("=> SYSU-MM loaded")
            self.print_dataset_statistics(train_RGB, train_IR, query, gallery[0])

        self.train_RGB  = train_RGB
        self.train_IR  = train_IR
        self.query = query
        self.gallery = gallery

        self.num_train_RGB_pids, self.num_train_RGB_imgs, self.num_train_RGB_cams = self.get_imagedata_info(self.train_RGB)
        self.num_train_IR_pids, self.num_train_IR_imgs, self.num_train_IR_cams = self.get_imagedata_info(self.train_IR)

        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery[0])

    def _get_pids(self, filename, path):
        with open(path + '/exp/' + filename + '.txt', 'r') as f:
            data = f.read()
            return map(int, data.split(','))

    def _process_dir(self, stage, modality, path, relabel=False, seed = 2019):
        random.seed(seed)

        pid_container = set()
        if stage == 'train':
            for id in self._get_pids('train_id', path):
                pid_container.add(id)
            for id in self._get_pids('val_id', path):
                pid_container.add(id)
        elif stage == 'test':
            for id in self._get_pids('test_id', path):
                pid_container.add(id)
        else:
            raise RuntimeError("'{}' is not available".format(stage))

        if modality == 'RGB':
            camlist = [0,1,3,4]
        elif modality == 'IR':
            camlist = [2,5]
        elif modality == 'indoor':
            camlist = [0,1]
        else:
            raise RuntimeError("'{}' is not available".format(modality))

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for i in camlist:
            campath = path+'/cam'+str(i+1)
            data = '*.jpg'
            for root, dirs, files in os.walk(campath, topdown=False):
                for name in dirs:
                    pid = int(name)
                    if(pid in pid_container):
                        # print(osp.join(campath, name, '*.jpg'))
                        img_paths = glob.glob(osp.join(campath, name, data))
                        if stage == 'test' and modality != 'IR':
                            single_shot = random.choice(img_paths)
                            dataset.append((single_shot, pid, i + 1))
                            # print(single_shot)
                        else:
                            for img_path in img_paths:
                                if relabel:
                                    label = pid2label[pid]
                                    dataset.append((img_path, label, i+1))
                                else:
                                    dataset.append((img_path, pid, i+1))

        return dataset

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

    def print_dataset_statistics(self, train_RGB, train_IR, query, gallery):
        num_train_RGB_pids, num_train_RGB_imgs = self.get_imagedata_info_r(train_RGB)
        num_train_IR_pids, num_train_IR_imgs = self.get_imagedata_info_r(train_IR)
        num_query_pids, num_query_imgs = self.get_imagedata_info_r(query)
        num_gallery_pids, num_gallery_imgs = self.get_imagedata_info_r(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images ")
        print("  ----------------------------------------")
        print("  train RGB| {:5d} | {:8d} ".format(num_train_RGB_pids, num_train_RGB_imgs))
        print("  train IR | {:5d} | {:8d} ".format(num_train_IR_pids, num_train_IR_imgs))
        print("  query    | {:5d} | {:8d} ".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d} ".format(num_gallery_pids, num_gallery_imgs))
        print("  ----------------------------------------")



