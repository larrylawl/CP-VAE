data_name=gyafc
path_to_checkpoint="/hdd2/lannliat/CP-VAE/checkpoint/vanilla-sharedweightsfrozen-z116-gyafc-fm/20220921-094434/"
# export CUDA_LAUNCH_BLOCKING=1

python transfer.py --data_name $data_name \
                   --load_path $path_to_checkpoint