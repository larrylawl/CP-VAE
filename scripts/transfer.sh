data_name=gyafc
path_to_checkpoint="/hdd2/lannliat/CP-VAE/checkpoint/subset-supervised-gyafc-fm/20220929-173443/"
export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1

python transfer.py --data_name $data_name \
                   --load_path $path_to_checkpoint