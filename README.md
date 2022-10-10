# RPP4 CP-VAE

## Setup
Create environment

```
conda env create -f env.yml
```

Set up the gyafc data directory. Data path follows this convention:

```
data_pth = os.path.join(args.hard_disk_dir, "data", args.data_name, "processed")
```

where `args.hard_disk_dir` and `args.data_name` are specified in the script.

## Training
Training with CP-VAE. Current model is CP-VAE with shared BERT encoder, GPT2 decoder, and style loss.

```
data_name=gyafc
save=checkpoint/exp_name
hard_disk_dir=/hdd2/lannliat/CP-VAE  # change to your data directory

python run.py --hard_disk_dir $hard_disk_dir
            --data_name $data_name \
            --save $save \
            --subset \  # trains with only subset of data for faster experiments
            --to_plot  # plots simplex p and umap of z1.
```

## Style Transfer
Apply style transfer on trained model. Currently `z1` is taken as the mean of the `z1` of the input validation examples.

```
data_name=gyafc
path_to_checkpoint="/hdd2/lannliat/CP-VAE/checkpoint/subset-cpvae-styleloss-gyafc-fm/20221003-091150/"

python transfer.py --data_name $data_name \
                   --load_path $path_to_checkpoint
```
