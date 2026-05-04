
DATASET=scientific

DEVICE="cuda:0"


python -u main.py \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 1024 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --e_dim 128 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 \
  --layers 512 256 \
  --vq_type vq \
  --loss_type mse \
  --dist l2 \
  --device $DEVICE \
  --kmeans_init True\
  --data_path ../dataset/${DATASET}/${DATASET}_emb_256.npy \
  --ckpt_dir ./rqvae_ckpt/${DATASET}
