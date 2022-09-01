data_name=gyafc
path_to_checkpoint="/hdd2/lannliat/CP-VAE/checkpoint/ours-valfixed-gyafc-glove/dup"

python transfer.py --data_name $data_name \
                   --load_path $path_to_checkpoint