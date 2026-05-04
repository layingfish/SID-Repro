import torch


def sampled_softmax(user_embedding: torch.tensor,
                    next_item: torch.tensor,
                    candidate_set: torch.tensor,
                    t: float = 1.0 ):


    target_embedding = torch.sum(next_item * user_embedding, dim=1, keepdim=True)
    product = torch.matmul(candidate_set, user_embedding[...,None]).squeeze(-1)
    loss = torch.exp(target_embedding / t) / (
                torch.sum(torch.exp(product / t), dim=1, keepdim=True) + torch.exp(target_embedding))
    loss = torch.mean(-torch.log(loss))
    return loss
