# RPP4 CP-VAE

## Setup
Create environment

```
conda env create -f env.yml
```

## Training
Training with CP-VAE. Current model is CP-VAE with shared encoder and style loss.

```
data_name=gyafc
save=checkpoint/exp_name

python run.py --data_name $data_name \
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