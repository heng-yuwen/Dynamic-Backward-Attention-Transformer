# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from torch.optim import Optimizer

from .lr_scheduler import LearningRateScheduler


class PolyLRScheduler(LearningRateScheduler):
    r"""
    Transformer Learning Rate Scheduler proposed in "Attention Is All You Need"

    Args:
        optimizer (Optimizer): Optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        final_lr (float): Final learning rate.
        final_lr_scale (float): Final learning rate scale
        warmup_steps (int): Warmup the learning rate linearly for the first N updates
        decay_steps (int): Steps in decay stages
    """
    def __init__(
            self,
            optimizer: Optimizer,
            power: float,
            num_epochs: int,
            final_lr: float,
            warmup_steps: int,
            by_epoch=True
    ) -> None:
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"

        super(PolyLRScheduler, self).__init__(optimizer, [pg["lr"] for pg in optimizer.param_groups])
        self.final_lr = final_lr # final lr after decay
        self.warmup_steps = warmup_steps
        self.update_steps = 1
        self.power = power
        self.num_epochs = num_epochs
        self.start_epoch = 0

    def _decide_stage(self):
        if self.update_steps <= self.warmup_steps:
            return 0
        else:
            return 1

    def step(self, epoch):
        stage = self._decide_stage()
        if stage == 0:
            warmup_rate = self.update_steps / self.warmup_steps
            self.set_lr(self.optimizer, warmup_rate)
            self.update_steps += 1
            if self.update_steps == self.warmup_steps:
                self.start_epoch = epoch
        elif stage == 1:  # start to decay with epoch
            decay_rate = (1-(epoch-self.start_epoch)/(self.num_epochs-self.start_epoch))**self.power
            self.set_lr(self.optimizer, decay_rate)
        else:
            raise ValueError("Undefined stage")
