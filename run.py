# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.dataset_utils import get_dataset
from utils.exp_utils import create_exp_dir, set_seed
from utils.text_utils import get_preprocessor
from transformers import AutoTokenizer
import argparse
import os
import torch
import time
import config
from models.decomposed_vae import DecomposedVAE
from sentence_transformers import SentenceTransformer
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

def main(args):
    set_seed(args.seed)
    start_time = time.time()
    conf = config.CONFIG[args.data_name]
    data_pth = os.path.join(args.hard_disk_dir, "data", args.data_name, "processed")
    enc_tokenizer = AutoTokenizer.from_pretrained(conf["params"]["vae_params"]["enc_name"])
    dec_tokenizer = AutoTokenizer.from_pretrained(conf["params"]["vae_params"]["dec_name"])
    # padding for gpt2: # https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16#training-script
    dec_tokenizer.pad_token = dec_tokenizer.unk_token  
    preprocessor_kwargs = {
        "data_dir": data_pth,
        "subset": args.subset,
    }
    preprocessor = get_preprocessor(args.data_name)(**preprocessor_kwargs)
    features = preprocessor.load_features(enc_tokenizer, dec_tokenizer, args.overwrite_cache)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = '{}-{}-{}'.format(args.save, args.data_name, args.feat)
    save_path = os.path.join(args.hard_disk_dir, save_path, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(save_path)
    scripts_to_save = [
        'run.py', 'models/decomposed_vae.py', 'models/vae.py',
        'models/base_network.py', 'models/bert_enc.py', 'config.py']
    logging = create_exp_dir(save_path, scripts_to_save=scripts_to_save,
                             debug=args.debug)

    ds = get_dataset(args.data_name)
    train_ds = ds(*features[0])
    dev_ds = ds(*features[1])
    test_ds = ds(*features[2])
    dl_params = {"batch_size": conf["bsz"],
                    "shuffle": True,
                    "drop_last": True} 
    train_dl = DataLoader(train_ds, **dl_params)
    dev_dl = DataLoader(dev_ds, **dl_params)
    test_dl = DataLoader(test_ds, **dl_params)

    kwargs = {
        "train": train_dl,
        "valid": dev_dl,
        "test": test_dl,
        "bsz": conf["bsz"],
        "save_path": save_path,
        "to_plot": args.to_plot,
        "logging": logging,
        "text_only": args.text_only,
        "writer": writer,
        "debug": args.debug,
    }
    params = conf["params"]
    params["vae_params"]["device"] = device
    if args.debug:
        params["num_epochs"] = 1
    kwargs = dict(kwargs, **params)

    model = DecomposedVAE(**kwargs)
    try:
        valid_loss = model.fit()
        logging("val loss : {}".format(valid_loss))
    except KeyboardInterrupt:
        logging("Exiting from training early")

    model.load(save_path)
    test_loss = model.evaluate(split="Test")
    logging("test loss: {}".format(test_loss))
    end_time = time.time() - start_time
    logging("total time taken: {}".format(end_time))

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',
                        help='data name')
    parser.add_argument('--hard_disk_dir', type=str, default='/hdd2/lannliat/CP-VAE')
    parser.add_argument('--save', type=str, default='checkpoint/ours',
                        help='directory name to save')
    parser.add_argument('--bsz', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--text_only', default=False, action='store_true',
                        help='use text only without feats. does not matter for our current iteration of cpvae')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode.')
    parser.add_argument('--subset', default=False, action='store_true', help="Use subset of training data for fast experimentation")
    parser.add_argument('--feat', type=str, default='fm',
                        help="feat repr. fm stands for foundation model."),
    parser.add_argument('--overwrite_cache', default=False, action="store_true")
    parser.add_argument('--to_plot', default=False, action="store_true", help="Plots simplex p and umap of z1.")
    parser.add_argument("--seed", type=int, default=888)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
