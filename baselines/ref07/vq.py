import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class RQVAE(nn.Module):
    def __init__(self, config, in_dim=768,):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.e_dim = config['e_dim']

        self.layers = config['layers']
        self.dropout_prob = config['dropout_prob']
        self.bn = config['bn']
        self.quant_loss_weight = config['alpha']
        self.beta = config['beta']
        self.vq_type = config['vq_type']

        if self.vq_type in ["vq"]:
            self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
            self.decode_layer_dims = self.encode_layer_dims[::-1]
        else:
            raise NotImplementedError


        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.rq = ResidualVectorQuantizer(config=config)
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x):
        latent = self.encoder(x)
        x_q, rq_loss, indices, code_one_hot, logit = self.rq(latent)
        out = self.decoder(x_q)

        return out, rq_loss, indices, code_one_hot, logit

    @torch.no_grad()
    def get_indices(self, xs):
        x_e = self.encoder(xs)
        indices = self.rq.get_indices(x_e)
        return indices

    @torch.no_grad()
    def get_maxk_indices(self, xs, maxk=1, used=False):

        x_e = self.encoder(xs)
        all_indices, fix = self.rq.get_maxk_indices(x_e, maxk=maxk, used=used)
        return all_indices, fix

    def get_codebook(self):
        return self.rq.get_codebook()


class ResidualVectorQuantizer(nn.Module):


    def __init__(self, config):
        super().__init__()
        self.n_e_list = config['num_emb_list']
        self.num_quantizers = len(self.n_e_list)
        self.vq_type = config['vq_type']
        self.dist = config['dist']
        if self.vq_type == "vq":
            self.vq_layers = nn.ModuleList([VectorQuantizer(config=config, n_e=n_e, dist=self.dist)
                                            for n_e in self.n_e_list])
        else:
            raise NotImplementedError

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook.detach().cpu())
        return torch.stack(all_codebook)

    @torch.no_grad()
    def get_indices(self, x):
        all_indices = []
        residual = x
        for i in range(len(self.vq_layers)):
            x_res, _, indices, _, _ = self.vq_layers[i](residual)
            residual = residual - x_res

            all_indices.append(indices)

        all_indices = torch.stack(all_indices, dim=-1)

        return all_indices

    def forward(self, x):
        all_losses = []
        all_indices = []
        all_one_hots = []
        all_logits = []

        x_q = 0
        x_q_detach = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, one_hot, logit = quantizer(residual)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_one_hots.append(one_hot)
            all_logits.append(logit)


        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_one_hots = torch.stack(all_one_hots, dim=1)
        all_logits = torch.stack(all_logits, dim=1)

        return x_q, mean_losses, all_indices, all_one_hots, all_logits


class VectorQuantizer(nn.Module):
    def __init__(self, config, n_e, dist):
        super().__init__()
        self.n_e = n_e
        self.dist = dist
        self.e_dim = config['e_dim']
        self.beta = config['beta']

        self.kmeans_init = config['kmeans_init']
        self.kmeans_iters = config['kmeans_iters']
        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        self.initted = False if self.kmeans_init else True
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):

        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, detach=True):

        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        if self.dist.lower() == 'l2':

            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())

        elif self.dist.lower() == 'dot':
            d = torch.matmul(latent, self.embedding.weight.t())
            d = -d
        elif self.dist.lower() == 'cos':
            d = torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding.weight, dim=-1).t())
            d = -d
        else:
            raise NotImplementedError


        indices = torch.argmin(d, dim=-1)
        code_one_hot = F.one_hot(indices, self.n_e).float()

        x_q = self.embedding(indices).view(x.shape)


        if self.dist.lower() == 'l2':
            codebook_loss = F.mse_loss(x_q, x.detach())
            commitment_loss = F.mse_loss(x_q.detach(), x)
            loss = codebook_loss + self.beta * commitment_loss
        elif self.dist.lower() in ['dot', 'cos']:
            loss = self.beta * F.cross_entropy(-d, indices.detach())
        else:
            raise NotImplementedError

        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        logit = d

        return x_q, loss, indices, code_one_hot, logit
