import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self, args, in_dim=768,):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.e_dim = args.e_dim

        self.layers = args.layers
        self.dropout_prob = args.dropout_prob
        self.bn = args.bn
        self.loss_type = args.loss_type
        self.quant_loss_weight= args.quant_loss_weight
        self.beta = args.beta
        self.vq_type = args.vq_type
        self.tau = 0.1

        if self.vq_type == "gumbel":
            self.h_dim = args.h_dim

        if self.vq_type == "vq" or self.vq_type == "ema":
            self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
            self.decode_layer_dims = self.encode_layer_dims[::-1]
        elif self.vq_type == "gumbel":
            self.encode_layer_dims = [self.in_dim] + self.layers + [self.h_dim]
            self.decode_layer_dims = [self.e_dim] + self.layers[::-1] + [self.in_dim]
        else:
            raise NotImplementedError


        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.rq = ResidualVectorQuantizer(args)
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, conflict=False):
        x_e = self.encoder(xs)
        indices = self.rq.get_indices(x_e, conflict=conflict)
        return indices

    @torch.no_grad()
    def get_maxk_indices(self, xs, maxk=1, used=False):

        x_e = self.encoder(xs)
        all_indices, fix = self.rq.get_maxk_indices(x_e, maxk=maxk, used=used)
        return all_indices, fix

    def get_codebook(self):
        return self.rq.get_codebook()

    @staticmethod
    def compute_contrastive_loss(query_embeds, semantic_embeds, temperature=0.07):
        gathered_query_embeds = query_embeds
        gathered_semantic_embeds = semantic_embeds
        gathered_query_embeds = F.normalize(gathered_query_embeds, dim=-1)
        gathered_semantic_embeds = F.normalize(gathered_semantic_embeds, dim=-1)
        effective_bsz = gathered_query_embeds.size(0)
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(gathered_query_embeds, gathered_semantic_embeds.transpose(0, 1)) / temperature

        co_loss = F.cross_entropy(similarities, labels)
        return co_loss

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        elif self.loss_type == 'infonce':
            loss_recon = self.compute_contrastive_loss(out, xs, temperature=self.tau)
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
