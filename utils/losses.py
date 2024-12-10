import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss * self.alpha)


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return torch.mean(focal_loss * self.alpha)


def dice_loss(input, target, gt=1, e=1e-4):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    # a = torch.sum(input * target, 1) # |X⋂Y|
    # b = torch.sum(input * input, 1) + 0.001  # |X|
    # c = torch.sum(target * target, 1) + 0.001  # |Y|
    # d = (2 * a) / (b + c)

    intersection = 2 * torch.sum(input * target, dim=1) + e
    union = torch.sum(input, dim=1) + torch.sum(target, dim=1) + e
    loss = 1 - intersection / union

    loss = torch.mean(loss)

    return loss

# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in
#         Focal Loss for Dense Object Detection.
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#         The losses are averaged across observations for each minibatch.
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
#                                    putting more focus on hard, misclassiﬁed examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#     """
#
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         # print(class_mask)
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P * class_mask).sum(1).view(-1, 1)
#
#         log_p = probs.log()
#         # print('probs size= {}'.format(probs.size()))
#         # print(probs)
#
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#         # print('-----bacth_loss------')
#         # print(batch_loss)
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss


class GHMC_loss(torch.nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmiod=True, loss_weight=1.0):
        super(GHMC_loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmiod
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight):
        '''

        :param pred:[batch_num, class_num]:
        :param target:[batch_num, class_num]:Binary class target for each sample.
        :param label_weight:[batch_num, class_num]: the value is 1 if the sample is valid and 0 if ignored.
        :return: GHMC_Loss
        '''
        if not self.use_sigmoid:
            raise NotImplementedError
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)
        valid = label_weight > 0
        total = max(valid.float().sum().item(), 1.0)
        n = 0  # the number of valid bins

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g <= edges[i + 1]) & valid
            num_in_bins = inds.sum().item()
            if num_in_bins > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bins
                    weights[inds] = total / self.acc_sum[i]
                else:
                    weights[inds] = total / num_in_bins
                n += 1

        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / total

        return loss * self.loss_weight

# class GHM_Loss(nn.Module):
#     def __init__(self, bins, alpha):
#         super(GHM_Loss, self).__init__()
#         self._bins = bins
#         self._alpha = alpha
#         self._last_bin_count = None
#
#     def _g2bin(self, g):
#         return torch.floor(g * (self._bins - 0.0001)).long()
#
#     def _custom_loss(self, x, target, weight):
#         raise NotImplementedError
#
#     def _custom_loss_grad(self, x, target):
#         raise NotImplementedError
#
#     def forward(self, x, target):
#         g = torch.abs(self._custom_loss_grad(x, target)).detach()
#
#         bin_idx = self._g2bin(g)
#
#         bin_count = torch.zeros((self._bins))
#         for i in range(self._bins):
#             bin_count[i] = (bin_idx == i).sum().item()
#
#         N = (x.size(0) * x.size(1))
#
#         if self._last_bin_count is None:
#             self._last_bin_count = bin_count
#         else:
#             bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
#             self._last_bin_count = bin_count
#
#         nonempty_bins = (bin_count > 0).sum().item()
#
#         gd = bin_count * nonempty_bins
#         gd = torch.clamp(gd, min=0.0001)
#         beta = N / gd
#         print(target.device)
#         print(bin_idx.device)
#         print(N.device)
#         print(gd.device)
#         print(beta.device)
#         print(beta[bin_idx].device)
#
#         return self._custom_loss(x, target, beta[bin_idx])
#
#
# class GHMC_Loss(GHM_Loss):
#     # 分类损失
#     def __init__(self, bins, alpha):
#         super(GHMC_Loss, self).__init__(bins, alpha)
#
#     def _custom_loss(self, x, target, weight):
#         return F.binary_cross_entropy_with_logits(x, target, weight=weight)
#
#     def _custom_loss_grad(self, x, target):
#         return torch.sigmoid(x).detach() - target
#
#
# class GHMR_Loss(GHM_Loss):
#     # 回归损失
#     def __init__(self, bins, alpha, mu):
#         super(GHMR_Loss, self).__init__(bins, alpha)
#         self._mu = mu
#
#     def _custom_loss(self, x, target, weight):
#         d = x - target
#         mu = self._mu
#         loss = torch.sqrt(d * d + mu * mu) - mu
#         N = x.size(0) * x.size(1)
#         return (loss * weight).sum() / N
#
#     def _custom_loss_grad(self, x, target):
#         d = x - target
#         mu = self._mu
#         return d / torch.sqrt(d * d + mu * mu)