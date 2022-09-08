# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.dataset_utils import get_dataset
from utils.exp_utils import create_exp_dir
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
    print("WARNING: currently only uses reg loss")
    start_time = time.time()
    conf = config.CONFIG[args.data_name]
    data_pth = os.path.join(args.hard_disk_dir, "data", args.data_name, "processed")
    enc_tokenizer = AutoTokenizer.from_pretrained(conf["params"]["vae_params"]["enc_name"])
    dec_tokenizer = AutoTokenizer.from_pretrained(conf["params"]["vae_params"]["dec_name"])
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    # padding for gpt2: # https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16#training-script
    dec_tokenizer.pad_token = dec_tokenizer.unk_token  
    preprocessor_kwargs = {
        "data_dir": data_pth,
        "enc_tokenizer": enc_tokenizer,
        "dec_tokenizer": dec_tokenizer,
        "overwrite_cache": args.overwrite_cache,
        "subset": args.subset,
        "sbert_model": sbert_model,
    }
    preprocessor = get_preprocessor(args.data_name)(**preprocessor_kwargs)
    features = preprocessor.load_features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = '{}-{}-{}'.format(args.save, args.data_name, args.feat)
    save_path = os.path.join(args.hard_disk_dir, save_path, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(save_path)
    scripts_to_save = [
        'run.py', 'models/decomposed_vae.py', 'models/vae.py',
        'models/base_network.py', 'config.py']
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

    # if args.text_only:
    #     train = train_data.create_data_batch(args.bsz, device)
    #     dev = dev_data.create_data_batch(args.bsz, device)
    #     test = test_data.create_data_batch(args.bsz, device)
    #     feat = train
    # else:
    #     train = train_data.create_data_batch_feats(args.bsz, train_feat, device)
    #     dev = dev_data.create_data_batch_feats(args.bsz, dev_feat, device)
    #     test = test_data.create_data_batch_feats(args.bsz, test_feat, device)
    #     feat = train_feat

    kwargs = {
        "train": train_dl,
        "valid": dev_dl,
        "test": test_dl,
        "bsz": conf["bsz"],
        "save_path": save_path,
        "logging": logging,
        "text_only": args.text_only,
        "writer": writer,
    }
    params = conf["params"]
    # params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
    params["vae_params"]["ni"] = train_ds.sent_embs[0].size(0)
    # params["vae_params"]["text_only"] = args.text_only
    # params["vae_params"]["mlp_ni"] = train_feat.shape[1]
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
                        help='use text only without feats')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode')
    parser.add_argument('--subset', default=False, action='store_true')
    parser.add_argument('--feat', type=str, default='fm',
                        help='feat repr')
    parser.add_argument('--overwrite_cache', default=False, action="store_true")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
