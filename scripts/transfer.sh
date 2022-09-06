data_name=gyafc
path_to_checkpoint="/hdd2/lannliat/CP-VAE/checkpoint/ours-10k-gyafc-fm/20220905-155102"

python transfer.py --data_name $data_name \
                   --load_path $path_to_checkpoint \
                   --subset \
                   --overwrite_cache