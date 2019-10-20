import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        # random.seed(seed)
        # np.random.seed(seed)

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class BalancedSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, dataset='RegDB', seed=2019):
        # random.seed(seed)
        # np.random.seed(seed)
        if dataset == 'SYSUMM':
            camlist = [1, 1, 0, 1, 1, 0]
        elif dataset == 'RegDB':
            camlist = [0, 1]
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // (self.num_instances*2)
        self.rgb_index_dic = defaultdict(list)
        self.ir_index_dic  = defaultdict(list)
        for index, (_, pid, camid) in enumerate(self.data_source):
            if camlist[camid-1] == 1:
                self.rgb_index_dic[pid].append(index)
            else:
                self.ir_index_dic[pid].append(index)
        self.pids = list(self.ir_index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            ir_idxs = self.ir_index_dic[pid]
            ir_num = len(ir_idxs)
            rgb_idxs = self.rgb_index_dic[pid]
            rgb_num = len(rgb_idxs)
            if rgb_num < ir_num:
                rgb_num = ir_num

            self.length += ir_num*2

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            ir_idxs = copy.deepcopy(self.ir_index_dic[pid])
            if len(ir_idxs) < self.num_instances:
                ir_idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(ir_idxs)

            rgb_idxs = copy.deepcopy(self.rgb_index_dic[pid])
            if len(rgb_idxs) < self.num_instances:
                rgb_idxs = np.random.choice(rgb_idxs, size=self.num_instances, replace=True)
            random.shuffle(rgb_idxs)

            ir_batch_idxs = []
            rgb_batch_idxs = []

            for idx in zip(ir_idxs,rgb_idxs):
                ir_batch_idxs.append(idx[0])
                rgb_batch_idxs.append(idx[1])
                if len(ir_batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(ir_batch_idxs+rgb_batch_idxs)
                    ir_batch_idxs = []
                    rgb_batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length