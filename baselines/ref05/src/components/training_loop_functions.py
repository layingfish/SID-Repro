import torch
from pytorch_lightning import LightningModule


def scale_loss_by_world_size_for_initialization_training_loop(
    model: LightningModule,
    loss: torch.Tensor,
    world_size: int,
    is_initialized: bool = True,
    initalization_optimizer_lr: float = 0.5,
    initalization_optimizer: torch.optim.Optimizer = torch.optim.SGD,
):


    if not is_initialized:


        opt = initalization_optimizer(model.parameters(), lr=initalization_optimizer_lr)
        loss = loss * world_size
    else:

        opt = model.optimizers()
    opt.zero_grad()
    model.manual_backward(loss)
    opt.step()
