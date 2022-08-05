data_name=yelp
path_to_target=/hdd2/lannliat/CP-VAE/checkpoint/ours-yelp-glove/20220802-134012/generated_results.txt

python evaluate.py --data_name $data_name \
                   --target_path $path_to_target