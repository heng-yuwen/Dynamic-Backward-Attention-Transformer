import torch
from pytorch_lightning.metrics import Metric


class ConfusionMatrix(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("confmat", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, confmat):
        self.confmat += confmat

    def compute(self):
        return self.confmat


class Acc(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, hist):
        self.correct += torch.diag(hist).sum().item()
        self.total += hist.sum().item()

    def compute(self):
        return self.correct.float() / self.total


class Loss(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, loss, count):
        self.loss += loss.item() * count
        self.total += count

    def compute(self):
        return self.loss.float() / self.total


class ClassAcc(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("hist", default=torch.zeros(23, 23, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, hist):
        self.hist += hist

    def compute(self):
        acc_cls = torch.diag(self.hist) / self.hist.sum(axis=1)
        return nanmean(acc_cls)


class MeanIU(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("hist", default=torch.zeros(23, 23, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, hist):
        self.hist += hist

    def compute(self):
        iu = torch.diag(self.hist) / (
            self.hist.sum(axis=1) + self.hist.sum(axis=0) - torch.diag(self.hist)
        )
        return nanmean(iu)


class FWIU(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("hist", default=torch.zeros(23, 23, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, hist):
        self.hist += hist

    def compute(self):
        iu = torch.diag(self.hist) / (
            self.hist.sum(axis=1) + self.hist.sum(axis=0) - torch.diag(self.hist)
        )
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwavacc


def nanmean(v, *args, inplace=True, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def nansum(v, *args, inplace=True, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs)