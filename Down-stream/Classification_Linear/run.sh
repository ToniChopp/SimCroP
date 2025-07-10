# CT-Rate
# SimCroP 1%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task CT-Rate --num_classes 18 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_CTRATE \
    --output_dir "output/CT-Rate/SimCroP/1/" --data_volume '1' --num_steps 1000  --eval_batch_size 64 \
    --learning_rate 1.5e-3 --warmup_steps 150 --fp16 --fp16_opt_level O2 --train_batch_size 32 \
    --patience 10
# SimCroP 10%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task CT-Rate --num_classes 18 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_CTRATE \
    --output_dir "output/CT-Rate/SimCroP/10/" --data_volume '10' --num_steps 10000  --eval_batch_size 64 \
    --learning_rate 3e-3 --warmup_steps 1500 --fp16 --fp16_opt_level O2 --train_batch_size 64 \
    --patience 10
# SimCroP 100%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task CT-Rate --num_classes 18 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_CTRATE \
    --output_dir "output/CT-Rate/SimCroP/100/" --data_volume '100' --num_steps 10000  --eval_batch_size 64 \
    --learning_rate 3e-3 --warmup_steps 1500 --fp16 --fp16_opt_level O2 --train_batch_size 64 \
    --patience 10

# RadChestCT
# SimCroP 10%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task RadChestCT --num_classes 16 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_RADCHESTCT \
    --output_dir "output/RadChestCT/SimCroP/10/" --data_volume '10' --num_steps 2000  --eval_batch_size 64 \
    --learning_rate 1.5e-3 --warmup_steps 80 --fp16 --fp16_opt_level O2 --train_batch_size 32 \
    --patience 10
# SimCroP 100%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task RadChestCT --num_classes 16 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_RADCHESTCT \
    --output_dir "output/RadChestCT/SimCroP/100/" --data_volume '100' --num_steps 2000  --eval_batch_size 128 \
    --learning_rate 3e-3 --warmup_steps 200 --fp16 --fp16_opt_level O2 --train_batch_size 128 \
    --patience 10


# CC-CCII
# SimCroP 1%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task CC-CCII --num_classes 3 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_CCII \
    --output_dir "output/CC-CCII/SimCroP/1/" --data_volume '1' --num_steps 200  --eval_batch_size 64 \
    --learning_rate 1.5e-3 --warmup_steps 20 --fp16 --fp16_opt_level O2 --train_batch_size 29 \
    --patience 20
# SimCroP 10%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task CC-CCII --num_classes 3 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_CCII \
    --output_dir "output/CC-CCII/SimCroP/10/" --data_volume '10' --num_steps 2000  --eval_batch_size 64 \
    --learning_rate 1.5e-3 --warmup_steps 150 --fp16 --fp16_opt_level O2 --train_batch_size 32 \
    --patience 20
# SimCroP 100%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task CC-CCII --num_classes 3 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_CCII \
    --output_dir "output/CC-CCII/SimCroP/100/" --data_volume '100' --num_steps 2000  --eval_batch_size 128 \
    --learning_rate 1.5e-3 --warmup_steps 200 --fp16 --fp16_opt_level O2 --train_batch_size 128 \
    --patience 20



# LUNA16
# simcrop 10%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task LUNA16 --num_classes 1 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_LUNA16 \
    --output_dir "output/LUNA16/SimCroP/10/" --data_volume '10' --num_steps 200  --eval_batch_size 64 \
    --learning_rate 1.5e-3 --warmup_steps 20 --fp16 --fp16_opt_level O2 --train_batch_size 31 \
    --patience 200
# simcrop 100%
CUDA_VISIBLE_DEVICES=0 python train.py --name simcrop --stage train --model vit_base_patch16 --task LUNA16 --num_classes 1 \
    --pretrained_path $PATH_TO_SimCroP_CHECKPOINT \
    --dataset_path $PATH_TO_LUNA16 \
    --output_dir "output/LUNA16/SimCroP/100/" --data_volume '100' --num_steps 2000  --eval_batch_size 128 \
    --learning_rate 1.5e-3 --warmup_steps 200 --fp16 --fp16_opt_level O2 --train_batch_size 32 \
    --patience 20

