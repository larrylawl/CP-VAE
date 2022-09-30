# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
import config
import torch
from utils.text_utils import MonoTextData
from models.decomposed_vae import DecomposedVAE
import argparse
import numpy as np
import os
from utils.dist_utils import cal_log_density
from torch.utils.data import DataLoader
from utils.dataset_utils import get_dataset
from utils.text_utils import get_preprocessor
from transformers import AutoTokenizer
from tqdm import tqdm
import random

def get_coordinates(a, b, p):
    pa = p - a
    ba = b - a
    t = torch.sum(pa * ba) / torch.sum(ba * ba)
    d = torch.norm(pa - t * ba, 2)
    return t, d

def process_transferred(text):
    # Regext pattern to match all newline characters
    pattern = "[\n|\r|\r\n]"
    processed = re.sub(pattern, ' ', text)
    return processed

def main(args):
    conf = config.CONFIG[args.data_name]
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
    ds = get_dataset(args.data_name)
    train_ds = ds(*features[0])
    dev_ds = ds(*features[1])
    test_ds = ds(*features[2])
    dl_params = {"batch_size": conf["bsz"],
                    "shuffle": True,
                    "drop_last": False} 
    train_dl = DataLoader(train_ds, **dl_params)
    dev_dl = DataLoader(dev_ds, **dl_params)
    test_dl = DataLoader(test_ds, **dl_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {
        "train": train_dl,
        "valid": dev_dl,
        "test": test_dl,
        "bsz": conf["bsz"],
        "save_path": args.load_path,
        "to_plot": None,
        "logging": None,
        "text_only": args.text_only,
        "writer": None,
        "debug": None,
    }

    params = conf["params"]
    params["vae_params"]["device"] = device
    kwargs = dict(kwargs, **params)

    model = DecomposedVAE(**kwargs)
    model.load(args.load_path)
    model.vae.eval()

    ### choosing mean z1  ###
    label_to_z1_mapping = {}
    label_to_p_mapping = {}
    for label in train_ds.labels_type:
        ds = train_ds.get_subset_specified_labels(label=label, nsamples=conf["bsz"])
        dl_params = {"batch_size": conf["bsz"],
                    "shuffle": False,
                    "drop_last": False} 
        dl = DataLoader(ds, **dl_params)
        batch, _ = next(iter(dl))
        enc_ids = batch["enc_input_ids"].to(device)
        enc_am = batch["enc_attention_mask"].to(device)
        style_labels = batch["labels"].to(device)
        z1, _, p = model.vae.enc.encode_semantic(enc_ids, enc_am, style_labels)
        z1 = z1.mean(0)
        p = p.mean(0)
        label_to_z1_mapping[label] = z1
        label_to_p_mapping[label] = p
    print(f"Label label_to_p_mapping: {label_to_p_mapping}")
    # ### choosing basis vector most representative of label  ###
    # label_to_simplex_mapping = {}
    # chosen_p_idx = set()
    # for label in train_ds.labels_type:
    #     ds = train_ds.get_subset_specified_labels(label=label, nsamples=conf["bsz"])
    #     dl_params = {"batch_size": conf["bsz"],
    #                 "shuffle": False,
    #                 "drop_last": False} 
    #     dl = DataLoader(ds, **dl_params)
    #     batch, _ = next(iter(dl))
    #     enc_ids = batch["enc_input_ids"].to(device)
    #     enc_am = batch["enc_attention_mask"].to(device)
    #     _, _, p = model.vae.enc.encode_semantic(enc_ids, enc_am)
    #     p = p.mean(0)
    #     topk = p.topk(params["vae_params"]["n_vars"], dim=0)
    #     for idx in topk[1]:
    #         idx = idx.item()
    #         if idx not in chosen_p_idx:
    #             chosen_p_idx.add(idx)
    #             label_to_simplex_mapping[label] = (idx, p)
    #             break
    #         print("Collision!!! Using next most positive.")
        
    #     assert label in label_to_simplex_mapping
    # print(f"Label to (basis, mean prob) mapping: {label_to_simplex_mapping}")

    # transfer to another label's basis vector that's NOT their own label
    dl_params = {"batch_size": conf["bsz"],
                    "shuffle": False,
                    "drop_last": False} 
    with open(os.path.join(args.load_path, 'generated_results.txt'), "w") as f:
        with torch.no_grad():
            # repeat for all label
            for label_type in test_ds.labels_type:
                ds = test_ds.get_subset_specified_labels(label_type, len(test_ds))
                dl = DataLoader(ds, **dl_params)
                for batch, _ in tqdm(dl):
                    enc_ids = batch["enc_input_ids"].to(device)
                    enc_am = batch["enc_attention_mask"].to(device)
                    dec_ids = batch["dec_input_ids"].to(device)
                    # z1, _, _ = model.vae.enc.encode_semantic(enc_ids, enc_am)
                    z2, _ = model.vae.enc.encode_syntax(enc_ids, enc_am)


                    shuffled_labels = random.sample(list(test_ds.labels_type), len(test_ds.labels_type))
                    for tra_label in shuffled_labels:
                        if tra_label != label_type:
                            tra_z1 = label_to_z1_mapping[tra_label]
                            # idx, p = label_to_simplex_mapping[tra_label]
                            break
                    # tra_z1 = model.vae.enc.var_embedding[idx, :].expand(z2.size(1), -1)
                    z = torch.cat([tra_z1, z2.squeeze()], -1)
                    context = dec_ids[:, 0].unsqueeze(-1)  # first word as context-
                    model.vae.dec.generate(context, max_length=50)
                    exit(1)
                    generated = model.vae.sample_sequence_conditional_batch(context=context ,past=z)
                    generated = dec_tokenizer.batch_decode(generated, skip_special_tokens=True)
                    # for debugging
                    pre_generated = enc_tokenizer.batch_decode(enc_ids, skip_special_tokens=True)
                    print(f"pregen: {pre_generated}")
                    print(f"gen: {generated}")
                    exit(1)
                    for text in generated:
                        f.write("%d\t%s\n" % (tra_label, process_transferred(text)))
                    break

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--hard_disk_dir', type=str, default='/hdd2/lannliat/CP-VAE')
    parser.add_argument('--feat', type=str, default='fm')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--text_only', default=False, action='store_true')
    parser.add_argument('--overwrite_cache', default=False, action="store_true")
    parser.add_argument('--subset', default=False, action='store_true')
    # parser.add_argument('--strategy', type=str, default='basis', help="(basis/mean)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
