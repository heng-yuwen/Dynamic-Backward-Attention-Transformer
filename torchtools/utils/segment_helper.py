from distutils.version import LooseVersion
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

a = np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
b = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
weisht_a = np.zeros((23, 23, 3, 3), dtype=np.float64)
weisht_a[range(23), range(23), :, :] = a
weisht_b = np.zeros((23, 23, 3, 3), dtype=np.float64)
weisht_b[range(23), range(23), :, :] = b

conv1 = nn.Conv2d(23, 23, kernel_size=3, stride=1, padding=1, bias=False)
conv2 = nn.Conv2d(23, 23, kernel_size=3, stride=1, padding=1, bias=False)
conv1.weight.data.copy_(torch.from_numpy(weisht_a).float())
conv2.weight.data.copy_(torch.from_numpy(weisht_b).float())
conv1.requires_grad = False
conv2.requires_grad = False


def boundary_loss(pred, target):
    # input: probability map
    # target: canny edge
    theta = 5

    p = F.softmax(pred, dim=1)
    p -= 0.2
    p = F.sigmoid(5 * p)

    p_a = conv1(p)
    p_b = conv2(p)
    p = torch.sqrt(torch.pow(p_a, 2) + torch.pow(p_b, 2))

    # n*1*500*500
    p = torch.sum(p, dim=1)
    p -= 0.2
    p = F.relu(5 * p)
    p = F.sigmoid(p)

    n, _, _ = p.shape

    gt_b_ext = F.max_pool2d(
        target, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    pred_b_ext = F.max_pool2d(
        p, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    gt_b = target.view(n, -1)
    pred_b = p.view(n, -1)
    gt_b_ext = gt_b_ext.view(n, -1)
    pred_b_ext = pred_b_ext.view(n, -1)

    # Precision, Recall
    P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
    R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

    # Boundary F1 Score
    BF1 = 2 * P * R / (P + R + 1e-7)

    loss = torch.mean(1 - BF1)

    return loss


def cross_entropy2d(pred, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = pred.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(pred)
    else:
        # >=0.3
        log_p = F.log_softmax(pred, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


def tensor_confusion_matrix(preds, label_trues, n_class, ignore=-1):
    label_preds = preds.argmax(dim=1)
    mask = (label_trues != ignore) & (label_trues < n_class)
    hist = torch.bincount(n_class * label_trues[mask] + label_preds[mask], minlength=n_class ** 2).reshape(n_class,
                                                                                                           n_class)

    return hist


def confusion_matrix(preds, label_trues, n_class, ignore=-1):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    is_list = isinstance(preds, list)
    if not is_list:
        return tensor_confusion_matrix(preds, label_trues, n_class, ignore)
    else:
        hist = []
        for idx in range(len(preds)):
            hist.append(tensor_confusion_matrix(preds[idx], label_trues[idx], n_class, ignore))
        hist = torch.stack(hist)
        hist = torch.sum(hist, dim=0)
    return hist


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask] +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
