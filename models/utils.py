'''
Misc Utility functions
'''

import os
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from utils.metrics import segmentation_scores, dice_score_list, single_class_dice_score, roc_auc
from sklearn import metrics
from .layers.loss import *

def get_optimizer(option, params):
    opt_alg = 'sgd' if not hasattr(option, 'optim') else option.optim
    if opt_alg == 'sgd':
        optimizer = optim.SGD(params,
                              lr=option.lr_rate,
                              momentum=0.9,
                              nesterov=True,
                              weight_decay=option.l2_reg_weight)

    if opt_alg == 'adam':
        optimizer = optim.Adam(params,
                               lr=option.lr_rate,
                               betas=(0.9, 0.999),
                               weight_decay=option.l2_reg_weight)

    return optimizer


def get_criterion(opts):
    if opts.criterion == 'cross_entropy':
        if opts.type == 'seg':
            criterion = cross_entropy_2D if opts.tensor_dim == '2D' else cross_entropy_3D
        elif 'classifier' in opts.type:
            criterion = CrossEntropyLoss()
    elif opts.criterion == 'dice_loss':
        criterion = SoftDiceLoss(opts.output_nc)
    elif opts.criterion == 'specific_classes_dice_loss':
        criterion = SelectClassSoftDiceLoss(opts.output_nc, class_ids=opts.loss_class_idx)
    elif opts.criterion == 'focal_tversky_loss':
        criterion = FocalTverskyLoss()
    elif opts.criterion == 'specific_classes_focal_tversky_loss':
        criterion = FocalTverskyLoss(class_ids=opts.loss_class_idx)
    elif opts.criterion == 'weighted_binary_cross_entropy_loss':
        criterion = WeightedBinaryCrossEntropyLoss(opts.output_nc)
    elif opts.criterion == 'l1_loss':
        criterion = L1Loss(opts.output_nc)
    elif opts.criterion == 'combined_loss':
        criterion = CombinedLoss(opts.output_nc)
    elif opts.criterion == 'single_class_combined_loss':
        criterion = CombinedLoss(opts.output_nc, class_id=1)

    return criterion

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - iter/max_iter)**power


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def segmentation_stats(prediction, target):
    n_classes = prediction.size(1)
    if n_classes == 1:
        pred_lbls = (torch.sigmoid(prediction) > 0.5)[0].int().cpu().numpy()
        n_unique_classes = n_classes + 1
    else:
        pred_lbls = prediction.data.max(1)[1].cpu().numpy()
        n_unique_classes = n_classes

    gt = np.squeeze(target.data.cpu().numpy(), axis=1)
    gts, preds = [], []
    for gt_, pred_ in zip(gt, pred_lbls):
        gts.append(gt_)
        preds.append(pred_)

    iou = segmentation_scores(gts, preds, n_class=n_unique_classes)
    class_wise_dice = dice_score_list(gts, preds, n_class=n_unique_classes)
    single_class_dice = single_class_dice_score(gts, preds)
    roc_auc_score = roc_auc(gts, preds)

    return iou, class_wise_dice, single_class_dice, roc_auc_score


def classification_scores(gts, preds, labels):
    accuracy        = metrics.accuracy_score(gts,  preds)
    class_accuracies = []
    for lab in labels: # TODO Fix
        class_accuracies.append(metrics.accuracy_score(gts[gts == lab], preds[gts == lab]))
    class_accuracies = np.array(class_accuracies)

    f1_micro        = metrics.f1_score(gts,        preds, average='micro')
    precision_micro = metrics.precision_score(gts, preds, average='micro')
    recall_micro    = metrics.recall_score(gts,    preds, average='micro')
    f1_macro        = metrics.f1_score(gts,        preds, average='macro')
    precision_macro = metrics.precision_score(gts, preds, average='macro')
    recall_macro    = metrics.recall_score(gts,    preds, average='macro')

    # class wise score
    f1s        = metrics.f1_score(gts,        preds, average=None)
    precisions = metrics.precision_score(gts, preds, average=None)
    recalls    = metrics.recall_score(gts,    preds, average=None)

    confusion = metrics.confusion_matrix(gts,preds, labels=labels)

    #TODO confusion matrix, recall, precision
    return accuracy, f1_micro, precision_micro, recall_micro, f1_macro, precision_macro, recall_macro, confusion, class_accuracies, f1s, precisions, recalls


def classification_stats(pred_seg, target, labels):
    return classification_scores(target, pred_seg, labels)


class EarlyStopper():
    def __init__(self, json_opts, verbose=False):
        self.patience = json_opts.patience if hasattr(json_opts, 'patience') else 10
        self.min_epochs = json_opts.min_epochs if hasattr(json_opts, 'min_epochs') else 100
        self.monitor = json_opts.monitor if hasattr(json_opts, 'monitor') else 'Seg_Loss'
        self.verbose = verbose

        self.index = 0
        self.should_stop_early = False
        self.is_improving = True
        # Todo - it would probably make more sense to keep track of these variables in the model itself
        self.best_loss = None
        self.best_epoch = None
        self.current_loss_total = 0
        self.current_loss_count = 0

        if self.verbose:
            print(f'Using early stopping with: {json_opts}')

    def update(self, losses):
        '''
        Early stopper should be updated upon every batch for validation
        :param losses: OrderedDict of losses by their name as referenced in monitor
        '''
        self.current_loss_total += losses[self.monitor]
        self.current_loss_count += 1

    def get_current_validation_loss(self):
        if self.current_loss_total is None:
            return None
        return self.current_loss_total / self.current_loss_count

    def interrogate(self, epoch):
        current_loss = self.get_current_validation_loss()

        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_epoch = epoch

        elif current_loss <= self.best_loss:
            self.index = 0
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.is_improving = True
            if self.verbose:
                print('current loss {} improved from {} at epoch {}'.format(current_loss, self.best_loss, self.best_epoch),
                      '-- idx_early_stopping = {} / {}'.format(self.index, self.patience))
        else:
            self.index += 1
            self.is_improving = False
            if self.verbose:
                print('current loss {} did not improve from {} at epoch {}'.format(current_loss, self.best_loss, self.best_epoch),
                      '-- idx_early_stopping = {} / {}'.format(self.index, self.patience))

        if self.index >= self.patience and epoch >= self.min_epochs:  # start early stopping after epoch 100
            print('-- early stopping')
            self.should_stop_early = True

        self.reset()

        return self.should_stop_early

    def reset(self):
        self.current_loss_total = 0
        self.current_loss_count = 0
