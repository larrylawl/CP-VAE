data_name=gyafc
path_to_target="/hdd2/lannliat/CP-VAE/checkpoint/ours-gyafc-glove/20220811-161904/generated_results.txt"

python evaluate.py --data_name $data_name \
                   --target_path $path_to_target