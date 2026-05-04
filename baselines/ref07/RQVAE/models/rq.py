import torch
import torch.nn as nn

from .vq import VectorQuantizer, GumbelVectorQuantizer, EMAVectorQuantizer


class ResidualVectorQuantizer(nn.Module):


    def __init__(self, args):
        super().__init__()
        self.n_e_list = args.num_emb_list
        self.num_quantizers = len(self.n_e_list)
        self.vq_type = args.vq_type

        if self.vq_type == "vq":
            self.sk_epsilons = args.sk_epsilons
            self.vq_layers = nn.ModuleList([VectorQuantizer(args=args, n_e=n_e, sk_epsilon=sk_epsilon)
                                            for n_e, sk_epsilon in zip(self.n_e_list, self.sk_epsilons)])
        elif self.vq_type == "ema":
            self.sk_epsilons = args.sk_epsilons
            self.vq_layers = nn.ModuleList([EMAVectorQuantizer(args=args, n_e=n_e, sk_epsilon=sk_epsilon)
                                            for n_e, sk_epsilon in zip(self.n_e_list, self.sk_epsilons)])
        elif self.vq_type == "gumbel":
            self.vq_layers = nn.ModuleList([GumbelVectorQuantizer(args=args, n_e=n_e) for n_e in self.n_e_list])
        else:
            raise NotImplementedError


    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook.detach().cpu())
        return torch.stack(all_codebook)

    @torch.no_grad()
    def get_indices(self, x, conflict=False):
        all_indices = []
        residual = x
        for i in range(len(self.vq_layers)):
            if conflict and i == len(self.vq_layers) - 1:
                x_res, _, indices = self.vq_layers[i](residual, conflict=True)
            else:
                x_res, _, indices = self.vq_layers[i](residual, conflict=False)
            residual = residual - x_res

            all_indices.append(indices)

        all_indices = torch.stack(all_indices, dim=-1)

        return all_indices

    @torch.no_grad()
    def get_maxk_indices(self, x, maxk=1, used=False):

        all_indices = []
        residual = x
        for i in range(len(self.vq_layers)):
            if i == len(self.vq_layers) - 1:
                indices, fix = self.vq_layers[i].get_maxk_indices(residual, maxk=maxk,used=used)
                x_res=0
            else:
                x_res, _, indices = self.vq_layers[i](residual, conflict=False)

            residual = residual - x_res
            all_indices.append(indices)

        all_indices = torch.stack(all_indices, dim=-1)

        return all_indices, fix

    def forward(self, x):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices
