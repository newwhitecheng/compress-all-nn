CHECKPOINT_DIR=/root/workspace/tfmodel/inception/checkpoint/model.ckpt
DATASET_DIR=/root/workspace/data/flowers
TRAIN_DIR=/tmp/train_logs

CUDA_VISIBLE_DEVICES=3 python /root/workspace/models/research/slim/train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_DIR} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --quantize_delay=1 \
    --ignore_missing_vars=True \
    --max_number_of_steps=1000
