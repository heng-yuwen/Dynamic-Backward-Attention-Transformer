import torch
from pytorch_lightning.metrics import Metric


class Loss(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor, target: torch.Tensor):

        self.loss += loss.item() * target.numel()
        self.total += target.numel()

    def compute(self):
        return self.loss.float() / self.total
