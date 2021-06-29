import os.path as osp
import configparser
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass

        if self.transform is not None:
            img = self.transform(img)


        return img, pid, camid, img_path

class SYSU_ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None ):
        self.dataset = dataset
        self.transform = transform
        self.RGB_cam = [1,2,4,5]
        self.fake_data_root = '/home/lyq/Desktop/dataset/Reid/SYSU-MM/fake/'
        self.grey_data_root = '/home/lyq/Desktop/dataset/Reid/SYSU-MM/grey/'
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        got_real_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_real_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_real_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        if self.transform is not None:
            img = self.transform(img)
#         if camid in self.RGB_cam:
#             got_fake_img = False
#             fake_img_path = self.fake_data_root + ''.join(img_path.split('/')[-3:]).split('.')[0] + '_fake_B.png'
#             if not osp.exists(fake_img_path):
#                 raise IOError("{} does not exist".format(fake_img_path))
#             while not got_fake_img:
#                 try:
#                     fake_img = Image.open(fake_img_path).convert('RGB')
#                     got_fake_img = True
#                 except IOError:
#                     print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#                     pass
#             if self.transform is not None:
#                 fake_img = self.transform(fake_img)
#             got_grey_img = False
#             grey_img_path = self.grey_data_root + '/'.join(img_path.split('/')[-3:]).split('.')[0] + '.png'
#             if not osp.exists(grey_img_path):
#                 raise IOError("{} does not exist".format(grey_img_path))
#             while not got_grey_img:
#                 try:
#                     grey_img = Image.open(grey_img_path).convert('RGB')
#                     got_grey_img = True
#                 except IOError:
#                     print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#                     pass
#             if self.transform is not None:
#                 grey_img = self.transform(grey_img)
#             return {'real': img, 'fake': fake_img, 'grey': grey_img}, pid, camid, {'real': img_path, 'fake': fake_img_path, 'grey': grey_img_path}
#         else:
        return {'real': img}, pid, camid, {'real': img_path}

class RegDB_ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.RGB_cam = [2]
        self.fake_data_root = '/home/lyq/Desktop/dataset/Reid/RegDB/fake/'
        self.grey_data_root = '/home/lyq/Desktop/dataset/Reid/RegDB/Visible_grey/'
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        # print(img_path, pid, camid)
        got_real_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_real_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_real_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        if self.transform is not None:
            img = self.transform(img)
#         if camid in self.RGB_cam:
#             got_fake_img = False
#             fake_img_path = self.fake_data_root + ''.join(img_path.split('/')[-2:]).split('.')[0] + '_fake_B.bmp'
#             if not osp.exists(fake_img_path):
#                 raise IOError("{} does not exist".format(fake_img_path))
#             while not got_fake_img:
#                 try:
#                     fake_img = Image.open(fake_img_path).convert('RGB')
#                     got_fake_img = True
#                 except IOError:
#                     print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#                     pass
#             if self.transform is not None:
#                 fake_img = self.transform(fake_img)
#             got_grey_img = False
#             grey_img_path = self.grey_data_root + '/'.join(img_path.split('/')[-2:]).split('.')[0] + '.png'
#             if not osp.exists(grey_img_path):
#                 raise IOError("{} does not exist".format(grey_img_path))
#             while not got_grey_img:
#                 try:
#                     grey_img = Image.open(grey_img_path).convert('RGB')
#                     got_grey_img = True
#                 except IOError:
#                     print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#                     pass
#             if self.transform is not None:
#                 grey_img = self.transform(grey_img)
#             return {'real': img, 'fake': fake_img, 'grey': grey_img}, pid, camid, {'real': img_path, 'fake': fake_img_path, 'grey': grey_img_path}
#         else:
        return {'real': img }, pid, camid, {'real': img_path }
