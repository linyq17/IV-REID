from .baseline import *


def build_model(config, num_classes):
    model = Baseline(num_classes=num_classes,
                     last_stride=int(config['MODEL']['LAST_STRIDE']),
                     model_path=config['MODEL']['PRETRAIN_PATH'],
                     model_name=config['MODEL']['NAME'],
                     using_fc_bias=config['MODEL']['FC_BIAS'],
                     using_norm=config['MODEL']['USING_NORM'],
                     norm_type=config['MODEL']['NORM_TYPE'])
    return model

def bulid_classifier(num_classes, dim=2048):
    ide_net = IDENet(dim, num_classes)

    return ide_net