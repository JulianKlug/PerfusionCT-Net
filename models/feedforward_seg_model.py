import os
from collections import OrderedDict
from torch.autograd import Variable
import utils.utils as util
from .base_model import BaseModel
from .networks import get_network
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import segmentation_stats, get_optimizer, get_criterion
from .networks.utils import HookBasedFeatureExtractor


class FeedForwardSegmentation(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain

        # initialize state
        self.best_validation_loss = None
        self.best_epoch = 0
        self.is_improving = False

        # define network input and output pars
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim

        # load/define networks
        self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                               in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                               tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample)
        if self.use_cuda: self.net = self.net.cuda()

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_S)

            # print the network details
            if kwargs.get('verbose', True):
                print('Network is initialized')
                print_network(self.net)

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input.size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                _input = _input.permute(0,4,1,2,3).contiguous().view(bs[0]*bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.cuda() if self.use_cuda else _input
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                # assert self.input.size() == self.target.size()

    def forward(self, split):
        if split == 'train':
            self.prediction = self.net(Variable(self.input))
            self.pred_seg = None
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input))
                # Apply a softmax and return a segmentation map
                if self.prediction.shape[1] > 1: # multiclass
                    self.logits = self.net.apply_argmax_softmax(self.prediction, dim=1)
                    self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1) # give each voxel the class index with max proba
                else:
                    self.logits = self.net.apply_argmax_softmax(self.prediction, dim=None)
                    self.pred_seg = (self.logits > 0.5).float()

    def backward(self):
        self.loss_S = self.criterion(self.prediction, self.target)
        self.loss_S.backward()

    def optimize_parameters(self):
        self.net.train()
        self.forward(split='train')

        self.optimizer_S.zero_grad()
        self.backward()
        self.optimizer_S.step()

    # This function updates the network parameters every "accumulate_iters"
    def optimize_parameters_accumulate_grd(self, iteration):
        accumulate_iters = int(2)
        if iteration == 0: self.optimizer_S.zero_grad()
        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        self.loss_S = self.criterion(self.prediction, self.target)

    def get_segmentation_stats(self):
        self.seg_scores, self.class_dice_score, self.overall_dice_score, self.roc_auc_score = segmentation_stats(self.prediction, self.target)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou']),
                     ('Overall_Dice', self.overall_dice_score), ('ROC_AUC', self.roc_auc_score)]
        for class_id in range(self.class_dice_score.size):
            seg_stats.append(('Class_{}'.format(class_id), self.class_dice_score[class_id]))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        return OrderedDict([('Seg_Loss', self.loss_S.data.item())
                            ])

    def update_validation_state(self, epoch):
        '''
        Update model state with best state
        :return: is_improving (boolean, True if model is improving), best validation loss and associated epoch
        '''
        self.is_improving = False
        if self.best_validation_loss is None or self.loss_S.data.item() < self.best_validation_loss:
            self.best_validation_loss = self.loss_S.data.item()
            self.best_epoch = epoch
            self.is_improving = True
        return self.is_improving, self.best_validation_loss, self.best_epoch

    def get_current_visuals(self):
        inp_img = util.tensor2im(self.input, 'img')
        target_img = util.tensor2im(self.target, 'lbl')
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img), ('target_S', target_img)])

    def get_current_volumes(self):
        def to_volume(a): return a.cpu().float().detach().numpy()
        output = self.prediction if self.pred_seg is None else self.pred_seg
        return OrderedDict([('output', to_volume(output)),
                            ('input', to_volume(self.input)),
                            ('target', to_volume(self.target))])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time (self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp/float(bsize), bp/float(bsize)

    # save current weights
    def save(self, network_label, epoch_label):
        self.save_network(self.net, network_label, epoch_label, self.gpu_ids)

    # delete saved weights
    def delete(self, network_label, epoch_label):
        save_filename = '{0:03d}_net_{1}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)


