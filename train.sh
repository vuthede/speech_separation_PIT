
# rm -rf runs/ckpt_28082021
# CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 40-60 python3 train.py \
#               --snapshot ckpt_28082021\
#               --data_file /home/ubuntu/vuthede/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/audio_database/dataset_train.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 100\
#               --lr 0.001\
#               --train_batchsize 8\
#               --resume ./ckpt_28082021/epoch_9.pth.tar\
#               --mode val\

# rm -rf runs/ckpt_29082021_PIT
# CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 40-60 python3 train.py \
#               --snapshot ckpt_289082021_PIT\
#               --data_file /home/ubuntu/vuthede/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 30\
#               --lr 0.001\
#               --train_batchsize 8\
#               --mode train\
#               --output_viz viz\


# rm -rf runs/ckpt_01092021_PIT_
# CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 61-80 python3 train.py \
#               --snapshot ckpt_01092021_PIT_\
#               --data_file /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 30\
#               --lr 0.001\
#               --train_batchsize 8\
#               --mode train\
#               --output_viz viz\


# rm -rf runs/ckpt_02092021_PIT_libri_smallmdodel
# CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 61-80 python3 train.py \
#               --snapshot ckpt_02092021_PIT_libri_smallmdodel\
#               --data_file /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 100\
#               --lr 0.001\
#               --train_batchsize 4\
#               --mode val\
#               --output_viz small\
#               --resume  ckpt_02092021_PIT_libri_smallmdodel/epoch_51.pth.tar 


# rm -rf runs/ckpt_02092021_PIT_avspeech_smallmdodel
# CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 61-80 python3 train.py \
#               --snapshot ckpt_02092021_PIT_avspeech_smallmdodel\
#               --data_file /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 100\
#               --lr 0.001\
#               --train_batchsize 4\
#               --mode train\
#               --output_viz smallavspeech\
            #   --resume  ckpt_02092021_PIT_libri_smallmdodel/epoch_51.pth.tar 


# rm -rf runs/ckpt_02092021_PIT_avspeech_smallmdodel_PIT
# CUDA_VISIBLE_DEVICES=6 taskset --cpu-list 40-60 python3 train.py \
#               --snapshot ckpt_02092021_PIT_avspeech_smallmdodel_PIT\
#               --data_file /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 100\
#               --lr 0.001\
#               --train_batchsize 4\
#               --mode val\
#               --output_viz test_yo\
#               --resume ckpt_02092021_PIT_avspeech_smallmdodel_PIT/epoch_99.pth.tar


# rm -rf runs/ckpt_02092021_PIT_avspeech_smallmdodel_PIT_inlcudephase
# CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 61-80 python3 train.py \
#               --snapshot ckpt_02092021_PIT_avspeech_smallmdodel_PIT_inlcudephase\
#               --data_file /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 100\
#               --lr 0.001\
#               --train_batchsize 4\
#               --mode val\
#               --output_viz smallavspeechPIT_phase\
#               --resume ckpt_02092021_PIT_avspeech_smallmdodel_PIT_inlcudephase/epoch_96.pth.tar



# rm -rf runs/ckpt_02092021_PIT_avspeech_smallmdodel_PIT_inlcudephase_wrong
# CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 61-80 python3 train.py \
#               --snapshot ckpt_02092021_PIT_avspeech_smallmdodel_PIT_inlcudephase_wrong\
#               --data_file /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 100\
#               --lr 0.001\
#               --train_batchsize 4\
#               --mode train\
#               --output_viz smallavspeechPIT_phase_wrong\


# rm -rf runs/ckpt_02092021_PIT_bigmdodel_nophase_1fc
# CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 40-60 python3 train.py \
#               --snapshot ckpt_02092021_PIT_bigmdodel_nophase_1fc\
#               --data_file /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 100\
#               --lr 0.001\
#               --train_batchsize 4\
#               --mode train\
#               --output_viz big1fc\


# rm -rf runs/ckpt_02092021_PIT_bigmdodel_nophase_3fc
# CUDA_VISIBLE_DEVICES=6 taskset --cpu-list 20-40 python3 train.py \
#               --snapshot ckpt_02092021_PIT_bigmdodel_nophase_3fc\
#               --data_file /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 100\
#               --lr 0.001\
#               --train_batchsize 4\
#               --mode train\
#               --output_viz big3fc\


# rm -rf runs/ckpt_02092021_PIT_libri_bigmdodel
# CUDA_VISIBLE_DEVICES=6 taskset --cpu-list 40-60 python3 train.py \
#               --snapshot ckpt_02092021_PIT_libri_bigmdodel\
#               --data_file /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_train.txt\
#               --data_file_val /home/ubuntu/vuthede/speech_separation/data/audio/audio_database/dataset_val.txt\
#               --data_dir /home/ubuntu/vuthede/audio_database\
#               --arch AudioOnlyModel\
#               --consin_lr_scheduler 1\
#               --max_num_epoch 100\
#               --lr 0.001\
#               --train_batchsize 8\
#               --mode train\
#               --output_viz big\
