###### Use Permutation Invariant Training Loss
# rm -rf runs/ckpt_02092021_PIT_avspeech_smallmdodel_PIT_inlcudephase
# CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 61-80 python3 train.py \
#               --snapshot ckpt_02092021_PIT_avspeech_smallmdodel_PIT_inlcudephase\
#               --data_dir ./MiniLibriMix\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 200\
#               --lr 0.001\
#               --train_batchsize 4\
#               --mode train\
#               --output_viz viz_outout\

###### Use pairwise Neg Sisdr Loss. It give better result
rm -rf runs/ckpt_02092021_sir
CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 61-80 python3 train.py \
              --snapshot ckpt_02092021_sir\
              --data_dir ./MiniLibriMix\
              --arch AudioOnlyModel\
              --consin_lr_scheduler 1\
              --max_num_epoch 200\
              --lr 0.001\
              --train_batchsize 4\
              --mode train\
              --output_viz viz_outout\
              --use_pairwise_neg_sisdr_loss_for_backward 1