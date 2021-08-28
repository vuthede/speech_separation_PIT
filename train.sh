CUDA_VISIBLE_DEVICE=0 taskset --cpu-list 0-6 python3 train.py \
              --snapshot ckpt_28082021\
              --data_file /home/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
              --data_dir /home/vuthede/speech_separation/data/audio/audio_database\
              --arch AudioOnlyModel\
              --consin_lr_scheduler 1\
              --max_num_epoch 100\
              --lr 0.001\
              --train_batchsize 2\
