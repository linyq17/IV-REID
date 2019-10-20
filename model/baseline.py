# encoding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
from .resnet import ResNet, BasicBlock, Bottleneck
from torch.autograd import Variable


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, model_name, pretrain_choice = 'imagenet',
                 using_fc_bias='no', using_norm='yes', norm_type='bn'):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        else:
            self.base.random_init()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if norm_type == 'bn':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
        elif norm_type == 'gn':
            self.bottleneck = nn.GroupNorm(2 , self.in_planes)

        if using_fc_bias == 'no':
            self.share_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.share_classifier.apply(weights_init_classifier)
        elif using_fc_bias == 'yes':
            self.share_classifier = nn.Linear(self.in_planes, self.num_classes)
            self.share_classifier.apply(weights_init_kaiming)
        self.using_norm = using_norm
    def forward(self, x):
        feat = self.base(x)
        feat = self.gap(feat)  # (b, 2048, 1, 1)
        feat = feat.view(feat.shape[0], -1)  # flatten to (bs, 2048)
        global_feat = feat
        if self.using_norm == 'yes':
            feat = self.bottleneck(feat)  # normalize for softmax
        if self.training:
            cls_score = self.share_classifier(feat)
            return cls_score, feat # global feature for triplet loss
        else:
            return feat, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def load_model_param(self, trained_path):
        param_dict = torch.load(trained_path)
        # print(param_dict.state_dict())
        for i in param_dict.state_dict():
            self.state_dict()[i].copy_(param_dict.state_dict()[i])

    def print_parameter(self):
        print('---------- Encoder initialized -------------')
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % ('Encoder', num_params / 1e6))
        print('-----------------------------------------------')

class IDENet(nn.Module):
    def __init__(self, in_planes=2048, num_classes=395):
        super(IDENet, self).__init__()
        # self.rgb_classifier = nn.Sequential(nn.Linear(in_planes, num_classes))
        self.rgb_bn = nn.BatchNorm1d(in_planes)
        self.rgb_fc1 = nn.Linear(in_planes, 512)
        self.rgb_fc2 = nn.Linear(512, num_classes)
        self.rgb_fc1.apply(weights_init_kaiming)
        self.rgb_fc2.apply(weights_init_kaiming)
        # self.rgb_classifier.apply(weights_init_kaiming)
        # self.ir_classifier = nn.Linear(in_planes, num_classes)
        self.ir_bn = nn.BatchNorm1d(in_planes)
        self.ir_fc1 = nn.Linear(in_planes, 512)
        self.ir_fc2 = nn.Linear(512, num_classes)
        self.ir_fc1.apply(weights_init_kaiming)
        self.ir_fc2.apply(weights_init_kaiming)
        # self.ir_classifier.apply(weights_init_kaiming)

    def forward(self, x, stage='fix'):
        if stage == 'fix':
            rgb_x = self.rgb_bn(x)
            rgb_x = self.rgb_fc1(rgb_x)
            rgb_score = self.rgb_fc2(rgb_x)
            # rgb_score = self.rgb_classifier(rgb_x)
            ir_x = self.ir_bn(x)
            ir_x = self.ir_fc1(ir_x)
            ir_score = self.ir_fc2(ir_x)
            # ir_score = self.ir_classifier(ir_x)
            return [rgb_score,rgb_x, ir_score,ir_x]
            # return [rgb_score,[], ir_score,[]]
        elif stage == 'train_rgb':
            rgb_x = self.rgb_bn(x)
            rgb_x = self.rgb_fc1(rgb_x)
            rgb_score = self.rgb_fc2(rgb_x)
            # rgb_score = self.rgb_classifier(rgb_x)

            return rgb_score, rgb_x
        elif stage == 'train_ir':
            ir_x = self.ir_bn(x)
            ir_x = self.ir_fc1(ir_x)
            ir_score = self.ir_fc2(ir_x)
            # ir_score = self.ir_classifier(ir_x)

            return ir_score, ir_x
