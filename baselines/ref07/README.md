# ETEGRec

This is the official PyTorch implementation for the paper:

> [Generative Recommender with End-to-End Learnable Item Tokenization](https://doi.org/10.1145/3726302.3729989)

## Overview

We propose **ETEGRec**, a novel **E**nd-**T**o-**E**nd **G**enerative **Rec**ommender that unifies item tokenization and generative recommendation into a cohesive framework. Built on a dual encoder-decoder architecture, ETEGRec consists of an item tokenizer and a generative recommender. To enable synergistic interaction between these components, we propose a recommendation-oriented alignment strategy, which includes two key optimization objectives: sequence-item alignment and preference-semantic alignment. These objectives tightly couple the learning processes of the item tokenizer and the generative recommender, fostering mutual enhancement. Additionally, we develop an alternating optimization technique to ensure stable and efficient end-to-end training of the entire framework.

![model](./asset/model.png)

## Requirements

```
torch==2.4.0+cu121
numpy
accelerate
faiss
tqdm
scikit-learn
transformers
```

## Dataset

You can download the SASRec embeddings, pretrained RQVAE weights and interaction data used in our paper from [Google Drive](https://drive.google.com/drive/folders/1KiPpB7uq7eFc4qB74cFOxhtY3H8nWgAI?usp=sharing) 


## RQVAE Pretrain
```shell
cd RQVAE
bash run_pretrain.sh
```

## Train

```shell
bash run.sh
```
