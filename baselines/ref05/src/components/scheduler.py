import math

import torch


class WarmupCosineSchedulerNonzeroMin(torch.optim.lr_scheduler.LambdaLR):


    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        scheduler_steps: int,
        min_ratio: float = 0.1,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        **kwargs,
    ):


        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.num_cycles = num_cycles
        super(WarmupCosineSchedulerNonzeroMin, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(
        self,
        step: int,
    ) -> float:

        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        if step <= self.scheduler_steps:

            decay_ratio = float(step - self.warmup_steps) / float(
                max(1, self.scheduler_steps - self.warmup_steps)
            )
            coeff = 0.5 * (
                1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * decay_ratio)
            )
            return max(self.min_ratio, self.min_ratio + coeff * (1 - self.min_ratio))
        else:
            return self.min_ratio
