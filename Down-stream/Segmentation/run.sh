# BTCV
# simcrop
CUDA_VISIBLE_DEVICES=0 python main.py --pretrained_path "../checkpoints/simcrop.pth" --output_dir "./output/BTCV/simcrop" --out_channels 14 \
    --space_x 1.5 --space_y 1.5 --space_z 3.0 \
    --task BTCV --data_dir $PATHTOBTCV --max_epochs 2000 --json_path "dataset.json"  \
    --batch_size 1 --sw_batch_size 1


# LUNA16
# simcrop 10%
CUDA_VISIBLE_DEVICES=0 python main.py --pretrained_path "../checkpoints/simcrop.pth" --output_dir "./output/LUNA16/simcrop" --out_channels 4 \
    --space_x 1.0 --space_y 1.0 --space_z 1.0 --json_path "dataset_10.json" \
    --task LUNA16 --data_dir $PATHTOLUNA16 --max_epochs 400  --val_every 20 \
    --batch_size 1 --sw_batch_size 1
# simcrop 100%
CUDA_VISIBLE_DEVICES=0 python main.py --pretrained_path "../checkpoints/simcrop.pth" --output_dir "./output/LUNA16/simcrop" --out_channels 4 \
    --space_x 1.0 --space_y 1.0 --space_z 1.0 --json_path "dataset_100.json" \
    --task LUNA16 --data_dir $PATHTOLUNA16 --max_epochs 50  --val_every 5 \
    --batch_size 1 --sw_batch_size 1
