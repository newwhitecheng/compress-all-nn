DATASET_DIR=/root/workspace/imagenet
TRAIN_DIR=/root/workspace/tfmodel/mobilenet/train

CUDA_VISIBLE_DEVICES=3 python /root/workspace/models/research/slim/train_image_classifier.py \
--train_dir=${TRAIN_DIR} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=imagenet \
--dataset_split_name=train \
--model_name=mobilenet_v1 \
--eval_image_size=224 \
--quantize_delay=10 \
--max_number_of_steps=20
