import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        if self.n_classes == 1:
            input = torch.sigmoid(input).view(batch_size, self.n_classes, -1)
            target = target.contiguous().view(batch_size, self.n_classes, -1).float()
        else:
            input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
            target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        print('nonooonon', score.shape, score)

        return score


class CustomSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, class_ids):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1]. 2D, 3D and potentially beyond (not tested)...
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W] or [B, 1, H, W, Z ... [
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    batch_size = logits.shape[0]

    if len(logits.shape) > 4: # 3D
        one_hot_encoder = One_Hot(num_classes).forward
        true_1_hot = one_hot_encoder(true)
        if num_classes == 1:
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            # probas = F.softmax(logits, dim=1)
            probas = F.softmax(logits, dim=1).view(batch_size, num_classes, -1)
            true_1_hot = one_hot_encoder(true).contiguous().view(batch_size, num_classes, -1)
        dims = 2

    else: # 2D
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1).long()]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1).long()]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        dims = (0,) + tuple(range(2, true.ndimension()))

    true_1_hot = true_1_hot.type(logits.type())
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)


class FocalTverskyLoss(nn.Module):
    # Goal: better segmentation loss for small spots
    # from https://arxiv.org/pdf/1810.07842.pdf
    def __init__(self, weight=None):
        super(FocalTverskyLoss, self).__init__()
        # Tversky loss variables
        self.alpha = 0.3
        self.beta = 0.7
        # focal TL vars
        self.gamma = 0.75

    def forward(self, logits, targets):
        tl = tversky_loss(targets, logits, self.alpha, self.beta, eps=1e-7)
        return torch.pow(tl, self.gamma)


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda()

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        epsilon = 1e-6
        batch_size = input.size(0)

        if self.n_classes == 1:
            input = torch.sigmoid(input).view(batch_size, self.n_classes, -1)
            target = target.contiguous().view(batch_size, self.n_classes, -1).float()
        else:
            input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
            target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        input_f = input.flatten()
        target_f = target.flatten()
        pos_sample = target_f.sum()
        neg_sample = target_f.numel() - pos_sample
        ratio = neg_sample / pos_sample
        ratio_one = ratio / (ratio + 1)
        ratio_zero = 1 / (ratio + 1)
        loss = -2 * torch.mean(ratio_one * target_f * torch.log(input_f + epsilon)
                             + ratio_zero * (1 - target_f) * torch.log(1 - input_f + epsilon))

        return loss

# todo this probably only works for a single class
class VolumeLoss(nn.Module):
    def __init__(self, n_classes, classId = 1):
        super(VolumeLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.classId = classId

    def forward(self, input, target):
        # epsilon = 1e-6
        batch_size = input.size(0)

        # if self.n_classes == 1:
        #     input = torch.sigmoid(input).view(batch_size, self.n_classes, -1)
        #     target = target.contiguous().view(batch_size, self.n_classes, -1).float()
        # else:
        #     input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        #     target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        # print('iiiiiii', target.shape, input.shape)

        target_f = target.flatten()
        input_seg_f = (F.softmax(input, dim=1)[:, self.classId].flatten() > 0.5).float()
        # input_seg_f = torch.max(input. max(1)[1].unsqueeze(1).view(-1) # give each voxel the class index with max proba
        # input_seg_f = input.view(-1) # give each voxel the class index with max proba
        # print('iiiiiii', target_f.shape, input_seg_f.shape)
        weight = 1 / (target_f.sum() + 1)
        # print('iiiiiii', target_f.sum() - input_seg_f.sum(), weight)
        difference = torch.abs(input_seg_f.sum() - target_f.sum()) * weight / 4
        # print('yyyyooo', difference, target_f.sum())
        return difference


def vol_diff(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = tf.cast(K.greater(K.flatten(y_pred), 0.5),'float32') # changed from just y_pred
    weight = 1 / (K.sum(y_true_f) + 1)  # weighted edditted by YAnnan 3/18. original value is 1/100000.
    difference = K.abs(K.sum(y_pred_f) - K.sum(y_true_f)) * weight
    return difference/4


class L1Loss(nn.Module):
    def __init__(self, n_classes):
        super(L1Loss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        loss = nn.L1Loss()
        batch_size = input.size(0)

        # Prepare for L1 loss
        if self.n_classes == 1:
            loss_input = torch.sigmoid(input).view(batch_size, self.n_classes, -1)
            loss_target = target.contiguous().view(batch_size, self.n_classes, -1).float()
        else:
            loss_input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
            loss_target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        # Calculate L1 loss
        return loss(loss_input, loss_target)

class CombinedLoss(nn.Module):
    def __init__(self, n_classes):
        super(CombinedLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, input, target):
        dice_loss = SoftDiceLoss(self.n_classes)
        volume_loss = VolumeLoss(self.n_classes)
        wbce_loss = WeightedBinaryCrossEntropyLoss(self.n_classes)
        l1_loss = L1Loss(self.n_classes)

        combined_loss = 0.5 * dice_loss(input, target) + volume_loss(input, target) + wbce_loss(input, target) + l1_loss(input, target)

        return combined_loss


#
# class DiceinCombinedLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(DiceinCombinedLoss, self).__init__()
#         self.one_hot_encoder = One_Hot(n_classes).forward
#         self.n_classes = n_classes
#
#     def forward(self, input, target):
#         smooth = 0.01
#         batch_size = input.size(0)
#
#         # Calculate Dice Score
#         if self.n_classes == 1:
#             dice_input = torch.sigmoid(input).view(batch_size, self.n_classes, -1)
#             dice_target = target.contiguous().view(batch_size, self.n_classes, -1).float()
#         else:
#             dice_input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
#             dice_target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
#         inter = torch.sum(dice_input * dice_target, 2) + smooth
#         union = torch.sum(dice_input, 2) + torch.sum(dice_target, 2) + smooth
#         dice_score = torch.sum(2.0 * inter / union)
#         dice_score = 1.0 - dice_score  # / (float(batch_size) * float(self.n_classes))
#
#         if dice_score > 1e7:
#             print("dice score went too high")
#
#         return 0.5 * dice_score
#
#
# class VolumeLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(CombinedLoss, self).__init__()
#         self.one_hot_encoder = One_Hot(n_classes).forward
#         self.n_classes = n_classes
#
#     def forward(self, input, target):
#         smooth = 0.01
#         batch_size = input.size(0)
#
#         # Prepare for Volume Loss
#         if self.n_classes == 1:
#             dice_input = torch.sigmoid(input).view(batch_size, self.n_classes, -1)
#             dice_target = target.contiguous().view(batch_size, self.n_classes, -1).float()
#         else:
#             dice_input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
#             dice_target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
#
#         # Calculate Volume Loss
#         try:
#             Volume_Loss = torch.abs(torch.sum(dice_input - dice_target)) / self.n_classes / N_plus.item()
#         except ZeroDivisionError:
#             Volume_Loss = 0.0
#             print("Exception raised : The number of positive voxels is equal to 0")
#
#         if Volume_Loss > 1e7:
#             print("Volume_Loss went too high")
#
#         return 0.5 * Volume_Loss

if __name__ == '__main__':
    depth=3
    batch_size=2
    encoder = One_Hot(depth=depth).forward
    y = Variable(torch.LongTensor(batch_size, 1, 1, 2 ,2).random_() % depth).cuda()  # 4 classes,1x3x3 img
    y_onehot = encoder(y)
    x = Variable(torch.randn(y_onehot.size()).float()).cuda()
    dicemetric = SoftDiceLoss(n_classes=depth)
    dicemetric(x,y)