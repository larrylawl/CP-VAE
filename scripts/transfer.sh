data_name=gyafc
path_to_checkpoint="/hdd2/lannliat/CP-VAE/checkpoint/ours-10k-annealling-nz-fixed-gyafc-fm/20220906-134821"
# export CUDA_LAUNCH_BLOCKING=1

python transfer.py --data_name $data_name \
                   --load_path $path_to_checkpoint \
                   --subset