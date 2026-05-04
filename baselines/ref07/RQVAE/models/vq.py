import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class VectorQuantizer(nn.Module):

    def __init__(self, args, n_e, sk_epsilon=0.003,):
        super().__init__()
        self.n_e = n_e
        self.e_dim = args.e_dim
        self.beta = args.beta
        self.dist = 'l2'
        self.kmeans_init = args.kmeans_init
        self.kmeans_iters = args.kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = args.sk_iters
        self.tau = 0.1

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if not self.kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

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

    @staticmethod
    def center_distance_for_constraint(distances):

        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, conflict=False):

        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)


        if self.dist.lower() == 'l2':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        elif self.dist.lower() == 'dot':
            d = torch.matmul(latent, self.embedding.weight.t()) / self.tau
            d = -d
        elif self.dist.lower() == 'cos':
            d = torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding.weight, dim=-1).t()) / self.tau
            d = -d
        else:
            raise NotImplementedError

        if self.sk_epsilon > 0 and (self.training or conflict):
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        if self.dist.lower() == 'l2':
            codebook_loss = F.mse_loss(x_q, x.detach())
            commitment_loss = F.mse_loss(x_q.detach(), x)
            loss = codebook_loss + self.beta * commitment_loss
        elif self.dist.lower() in ['dot', 'cos']:
            d = - torch.matmul(F.normalize(latent.detach(), dim=-1), F.normalize(self.embedding.weight, dim=-1).t()) / self.tau
            loss = self.beta * F.cross_entropy(-d, indices.detach())
        else:
            raise NotImplementedError


        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

    @torch.no_grad()
    def get_maxk_indices(self, x, maxk=1, used=False):

        latent = x.view(-1, self.e_dim)

        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())

        d = -d
        topk_prob, topk_idx = d.topk(maxk + 1, dim=-1)

        if used:
            indices = topk_idx[:, maxk]
            fix = torch.zeros_like(indices, dtype=torch.bool)
        else:
            fix = (topk_prob[:, maxk-1] == topk_prob[:, maxk-1].max())

            indices = torch.where(fix, topk_idx[:, maxk-1], topk_idx[:, maxk])


        indices = indices.view(x.shape[:-1])

        return indices, fix

class EMAVectorQuantizer(nn.Module):

    def __init__(self, args, n_e, sk_epsilon=0.003,):
        super().__init__()
        self.n_e = n_e
        self.e_dim = args.e_dim
        self.beta = args.beta
        self.kmeans_init = args.kmeans_init
        self.kmeans_iters = args.kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = args.sk_iters
        self.decay = args.moving_avg_decay

        embedding = torch.randn(self.n_e, self.e_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('embedding_avg', embedding.clone())
        self.register_buffer('cluster_size', torch.ones(n_e))
        if not self.kmeans_init:
            self.initted = True
        else:
            self.initted = False

    def get_codebook(self):
        return self.embedding

    def get_codebook_entry(self, indices, shape=None):

        z_q = F.embedding(indices, self.embedding)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):

        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def _tile(self, x):
        n, d = x.shape
        if n < self.n_e:
            n_repeats = (self.n_e + n - 1) // n
            std = 0.01 / np.sqrt(d)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def forward(self, x, conflict=False):

        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)


        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.t())

        if self.sk_epsilon > 0 and (self.training or conflict):
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

        x_q = F.embedding(indices, self.embedding).view(x.shape)

        if self.training:
            embedding_onehot = F.one_hot(indices, self.n_e).type(latent.dtype)
            embedding_sum = embedding_onehot.t() @ latent
            moving_average(self.cluster_size, embedding_onehot.sum(0), self.decay)
            moving_average(self.embedding_avg, embedding_sum, self.decay)
            n = self.cluster_size.sum()
            cluster_size = laplace_smoothing(self.cluster_size, self.n_e) * n
            embedding_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embedding_normalized)

            temp = self._tile(latent)
            temp = temp[torch.randperm(temp.size(0))][:self.n_e]
            usage = (self.cluster_size.view(self.n_e, 1) >= 1).float()
            self.embedding.data.mul_(usage).add_(temp * (1 - usage))


        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = 0
        loss = codebook_loss + self.beta * commitment_loss


        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

    @torch.no_grad()
    def get_maxk_indices(self, x, maxk=1, used=False):

        latent = x.view(-1, self.e_dim)

        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.t())

        d = -d
        topk_prob, topk_idx = d.topk(maxk + 1, dim=-1)

        if used:
            indices = topk_idx[:, maxk]
            fix = torch.zeros_like(indices, dtype=torch.bool)
        else:
            fix = (topk_prob[:, maxk - 1] == topk_prob[:, maxk - 1].max())

            indices = torch.where(fix, topk_idx[:, maxk - 1], topk_idx[:, maxk])

        indices = indices.view(x.shape[:-1])

        return indices, fix

class GumbelVectorQuantizer(nn.Module):

    def __init__(self, args, n_e):
        super().__init__()
        self.n_e = n_e
        self.e_dim = args.e_dim
        self.h_dim = args.h_dim
        self.tau = args.temperature

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.proj = nn.Linear(self.h_dim, self.n_e, bias=False)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):

        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def forward(self, x, conflict=False):

        latent = x.view(-1, self.h_dim)

        logits = self.proj(latent)

        if self.training or conflict:
            soft_onehot = F.gumbel_softmax(logits, tau=self.tau, dim=-1, hard=False)
        else:
            soft_onehot = F.softmax(logits, dim=-1)

        indices = soft_onehot.argmax(dim=-1)

        x_q = torch.matmul(soft_onehot, self.embedding.weight)

        log_logits = F.log_softmax(logits, dim=-1)
        log_uniform = torch.full_like(log_logits, -torch.log(torch.tensor(self.n_e)))
        loss = F.kl_div(log_logits, log_uniform, reduction="batchmean", log_target=True)

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

    @torch.no_grad()
    def get_maxk_indices(self, x, maxk=1, used=False):

        latent = x.view(-1, self.h_dim)

        logits = self.proj(latent)

        soft_onehot = F.softmax(logits, dim=-1)

        topk_prob, topk_idx = soft_onehot.topk(maxk + 1, dim=-1)
        if used:
            indices = topk_idx[:, maxk]
            fix = torch.zeros_like(indices, dtype=torch.bool)
        else:
            fix = (topk_prob[:, maxk-1] == topk_prob[:, maxk-1].max())

            indices = torch.where(fix, topk_idx[:, maxk-1], topk_idx[:, maxk])


        indices = indices.view(x.shape[:-1])

        return indices, fix
