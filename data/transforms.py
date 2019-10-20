import torchvision.transforms as T
import math

def build_transforms(config, is_train = True ):

    if is_train:
        transform = T.Compose([
            T.Resize(list(map(int,config['DATA'].getlist('SIZE_TRAIN')))),
            T.RandomHorizontalFlip(p=float(config['DATA']['FLIP_PROB'])),
            T.Pad(int(config['DATA']['PADDING'])),
            T.RandomCrop(list(map(int,config['DATA'].getlist('SIZE_TRAIN')))),
            T.ToTensor(),
            T.Normalize(mean=list(map(float,config['DATA'].getlist('NORMALIZATION_MEAN'))),
                        std =list(map(float,config['DATA'].getlist('NORMALIZATION_STD')))
                        )
        ])
    else:
        transform = T.Compose([
            T.Resize(list(map(int, config['DATA'].getlist('SIZE_TEST')))),
            T.ToTensor(),
            T.Normalize(mean=list(map(float,config['DATA'].getlist('NORMALIZATION_MEAN'))),
                        std =list(map(float,config['DATA'].getlist('NORMALIZATION_STD')))
                        ),
        ])

    return transform