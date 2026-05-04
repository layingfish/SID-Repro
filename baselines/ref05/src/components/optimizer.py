from torch.optim import Optimizer


class PassThroughOptimizer(Optimizer):


    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(PassThroughOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        return None

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def state_dict(self):

        return {}

    def load_state_dict(self, state_dict):

        pass
