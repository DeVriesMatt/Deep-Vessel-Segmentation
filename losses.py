# Pywick losses

import torch
import torch.nn as nn
import torch.nn.functional as F

"""Adapted from https://github.com/achaiah/pywick/blob/master/pywick/losses.py"""


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, **kwargs):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, **kwargs):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # print('logits: {}, targets: {}'.format(logits.size(), targets.size()))
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        # smooth = 1.

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        # print(score.shape)
        # print(score)
        # score = score.reshape(num, 1)
        # print(score.shape)
        return score


class FocalLoss(nn.Module):
    """
    Weighs the contribution of each sample to the loss based in the classification error.
    If a sample is already classified correctly by the CNN, its contribution to the loss decreases.

    :eps: Focusing parameter. eps=0 is equivalent to BCE_loss
    """
    def __init__(self, l=0.5, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.l = l
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.view(-1)
        probs = torch.sigmoid(logits).view(-1)

        losses = -(targets * torch.pow((1. - probs), self.l) * torch.log(probs + self.eps) + \
                   (1. - targets) * torch.pow(probs, self.l) * torch.log(1. - probs + self.eps))
        loss = torch.mean(losses)

        return loss


class BCEDiceFocalLoss(nn.Module):
    '''
        :param num_classes: number of classes
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                            focus on hard misclassified example
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        :param weights: (list(), default = [1,1,1]) Optional weighing (0.0-1.0) of the losses in order of [bce, dice, focal]
    '''
    def __init__(self, focal_param=0.5, weights=[1.0,1.0,1.0], **kwargs):
        super(BCEDiceFocalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
        self.dice = SoftDiceLoss()
        self.focal = FocalLoss(l=focal_param)
        self.weights = weights

    def forward(self, logits, targets):
        # logits = logits.squeeze()
        return self.weights[0] * self.bce(logits, targets) + self.weights[1] * self.dice(logits, targets) + self.weights[2] * self.focal(logits.unsqueeze(1), targets.unsqueeze(1))


class BinaryFocalLoss(nn.Module):
    '''
        Implementation of binary focal loss. For multi-class focal loss use one of the other implementations.

        gamma = 0 is equivalent to BinaryCrossEntropy Loss
    '''
    def __init__(self, gamma=1.333, eps=1e-6, alpha=1.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class ComboBCEDiceLoss(nn.Module):
    """
        Combination BinaryCrossEntropy (BCE) and Dice Loss with an optional running mean and loss weighing.
    """

    def __init__(self, use_running_mean=False, bce_weight=1, dice_weight=1, eps=1e-6, gamma=0.9, combined_loss_only=True, **kwargs):
        """

        :param use_running_mean: - bool (default: False) Whether to accumulate a running mean and add it to the loss with (1-gamma)
        :param bce_weight: - float (default: 1.0) Weight multiplier for the BCE loss (relative to dice)
        :param dice_weight: - float (default: 1.0) Weight multiplier for the Dice loss (relative to BCE)
        :param eps: -
        :param gamma:
        :param combined_loss_only: - bool (default: True) whether to return a single combined loss or three separate losses
        """

        super().__init__()
        '''
        Note: BCEWithLogitsLoss already performs a torch.sigmoid(pred)
        before applying BCE!
        '''
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

        self.dice_weight = dice_weight # changed to 2
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma
        self.combined_loss_only = combined_loss_only

        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def to(self, device):
        super().to(device=device)
        self.bce_logits_loss.to(device=device)

    def reset_parameters(self):
        self.running_bce_loss.zero_()
        self.running_dice_loss.zero_()

    def forward(self, outputs, targets):
        # inputs and targets are assumed to be BxCxWxH (batch, color, width, height)
        # outputs = outputs.squeeze()       # necessary in case we're dealing with binary segmentation (color dim of 1)
        # print(len(outputs.shape), len(targets.shape))
        # assert len(outputs.shape) == len(targets.shape)
        # assert that B, W and H are the same
        assert outputs.size(-0) == targets.size(-0)
        assert outputs.size(-1) == targets.size(-1)
        assert outputs.size(-2) == targets.size(-2)

        bce_loss = self.bce_logits_loss(outputs, targets)

        dice_target = (targets == 1).float()
        dice_output = torch.sigmoid(outputs)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + self.eps
        dice_loss = (-torch.log(2 * intersection / union))

        if self.use_running_mean == False:
            bmw = self.bce_weight
            dmw = self.dice_weight
            # loss += torch.clamp(1 - torch.log(2 * intersection / union),0,100)  * self.dice_weight
        else:
            self.running_bce_loss = self.running_bce_loss * self.gamma + bce_loss.data * (1 - self.gamma)
            self.running_dice_loss = self.running_dice_loss * self.gamma + dice_loss.data * (1 - self.gamma)

            bm = float(self.running_bce_loss)
            dm = float(self.running_dice_loss)

            bmw = 1 - bm / (bm + dm)
            dmw = 1 - dm / (bm + dm)

        loss = bce_loss * bmw + dice_loss * dmw

        if self.combined_loss_only:
            return loss
        else:
            return loss, bce_loss, dice_loss
