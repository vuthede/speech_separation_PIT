rm -rf runs/ckpt_02092021_PIT_avspeech_smallmdodel_PIT_inlcudephase
CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 61-80 python3 train.py \
              --snapshot ckpt_02092021_PIT_avspeech_smallmdodel_PIT_inlcudephase\
              --data_dir ./MiniLibriMix\
              --arch AudioOnlyModel\
              --consin_lr_scheduler 1\
              --max_num_epoch 200\
              --lr 0.001\
              --train_batchsize 4\
              --mode train\
              --output_viz smallavspeechPIT_phase\