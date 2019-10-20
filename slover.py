import torch
from bisect import bisect_right
from layers import TripletLoss


# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, config, optimizer, last_epoch=-1):

        milestones = list(map(int, config['TRAIN'].getlist('STEPS')))
        gamma = float(config['TRAIN']['GAMMA'])
        warmup_factor = float(config['TRAIN']['WARMUP_FACTOR'])
        warmup_iters = int(config['TRAIN']['WARMUP_ITERS'])
        warmup_method = config['TRAIN']['WARMUP_METHOD']
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def make_optimizer(config, model,net='backbone'):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = float(config['TRAIN']['BASE_LR'])
        if net == 'head':
            lr = lr*10
        weight_decay = float(config['TRAIN']['WEIGHT_DECAY'])
        # if "bias" in key:
        #     lr = float(config['TRAIN']['BASE_LR']) * float(config['TRAIN']['BIAS_LR_FACTOR'])
        #     weight_decay = float(config['TRAIN']['WEIGHT_DECAY_BIAS'])
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # print(model)
    if config['TRAIN']['OPTIMAZER'] == 'SGD':
        optimizer = getattr(torch.optim, config['TRAIN']['OPTIMAZER'])(params, momentum=float(config['TRAIN']['MOMENTUM']))
    else:
        optimizer = getattr(torch.optim, config['TRAIN']['OPTIMAZER'])(params)
    return optimizer

def make_loss(config, num_classes):
    id_loss = torch.nn.CrossEntropyLoss()
    triplet = TripletLoss(float(config['TRAIN']['MARGIN']))
    def id_loss_func(score, feat, target):
        return id_loss(score, target)
    def triplet_loss_func(feat, target):
        return triplet(feat, target)[0]

    return (id_loss_func, triplet_loss_func)