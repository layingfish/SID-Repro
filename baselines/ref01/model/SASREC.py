import numpy as np
import torch
import torch.nn as nn
from .inputs import item_feat_no_add_feat as item_feat

from .module_utils import sampled_softmax


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASREC(torch.nn.Module):


    def __init__(self, config):
        super(SASREC, self).__init__()

        self.num_neg = config['num_neg']

        self.item_feat = item_feat(config)
        config['hidden_dim'] = self.item_feat.size

        self.pos_emb = torch.nn.Embedding(config['max_reco_his'], config['hidden_dim'])
        self.emb_dropout = torch.nn.Dropout(p = config['dropout_rate'])

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(config['hidden_dim'], eps=1e-8)

        for _ in range(config['num_blocks']):
            new_attn_layernorm = torch.nn.LayerNorm(config['hidden_dim'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(config['hidden_dim'],
                                                            config['num_heads'],
                                                            config['dropout_rate'])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(config['hidden_dim'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config['hidden_dim'], config['dropout_rate'])
            self.forward_layers.append(new_fwd_layer)

        self.sigmoid = torch.nn.Sigmoid()

        self.loss_func = torch.nn.BCELoss()

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m)


    def log2feats(self, seqs, log_seqs_mask):

        seqs *= seqs.size(-1) ** 0.5
        positions = np.tile(np.array(range(log_seqs_mask.shape[1])), [log_seqs_mask.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs_mask
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)


            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        timeline_mask = torch.sum(~timeline_mask, dim=-1) - 1
        log_feats = self.gather_indexes(log_feats, timeline_mask)
        return log_feats

    def gather_indexes(self, output, gather_index):

        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def parse_input_test(self, input_data):

        user, rec_his = input_data

        rec_his_emb = self.item_feat(rec_his)
        rec_his_mask = torch.where(
                            rec_his==0,
                            1, 0).bool()

        return rec_his_emb, rec_his_mask


    def parse_input_train(self, input_data):

        user, rec_his, pos_item = input_data


        neg_item = self._neg_sample_per_batch(pos_item.device)
        neg_item = neg_item.unsqueeze(dim=0).expand(pos_item.size(0), -1)

        pos_item_emb = self.item_feat(pos_item)
        neg_item_emb = self.item_feat(neg_item)

        rec_his_emb = self.item_feat(rec_his)
        rec_his_mask = torch.where(
                            rec_his==0,
                            1, 0).bool()

        return pos_item_emb, neg_item_emb, rec_his_emb, rec_his_mask


    def predict(self, input_data, topK = 50):


        rec_his_emb, rec_his_mask = self.parse_input_test(input_data)


        log_feats = self.log2feats(rec_his_emb, rec_his_mask)


        all_item_emb = self.item_feat(
            torch.arange(1, self.item_feat.item_num, device=log_feats.device))


        logits = self.sigmoid( torch.matmul(log_feats, torch.t(all_item_emb)))
        logit_value, logit_index = torch.topk(logits, k=topK, dim=1)
        logit_index += 1

        return logit_value, logit_index

    def _neg_sample_per_batch(self, device):
        numbers = np.arange(1, self.item_feat.item_num)

        unique_IDs = np.random.choice(numbers, size=self.num_neg, replace=False)
        unique_IDs = torch.tensor(unique_IDs, dtype=torch.long, device=device)

        return unique_IDs


    def forward(self, input_data):

        pos_item_emb, neg_item_emb, rec_his_emb, rec_his_mask = self.parse_input_train(input_data)

        log_feats = self.log2feats(rec_his_emb, rec_his_mask)

        loss = sampled_softmax(log_feats, pos_item_emb, neg_item_emb)

        return loss
