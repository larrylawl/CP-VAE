data_name=yelp
hdd='/hdd2/lannliat/CP-VAE'
path_to_checkpoint=$hdd/checkpoint/ours-yelp-glove/20220802-134012

python transfer.py --data_name $data_name \
                   --load_path $path_to_checkpoint