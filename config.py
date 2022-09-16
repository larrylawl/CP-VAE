# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ------------------------- PATH -----------------------------
ROOT_DIR = "."
DATA_DIR = "%s/data" % ROOT_DIR

# ------------------------- DATA -----------------------------
CONFIG = {}
CONFIG["gyafc"] = {
    "ref0": "gyafc_all_model_prediction_ref0.csv",
    "ref1": "gyafc_all_model_prediction_ref1.csv",
    "bsz": 32,
    "label": True,
    "params": {
        "log_interval": 2000,
        "num_epochs": 10,
        "enc_lr": 5e-5,
        "dec_lr": 5e-5,
        "warm_up": 10,
        "kl_start": 0,
        "beta1": 0,  # 0 to use annealing kl weight
        "beta2": 0,  # 0 to use annealing kl weight
        "srec_weight": 2.5, # original: 1
        "reg_weight": 10, # original: 1
        "ic_weight": 0.0,
        "aggressive": False,
        "vae_params": {
            "syn_nz": 40,
            "sem_nz": 80,
            "n_vars": 3,
            "enc_name": "distilbert-base-uncased",
            "dec_name": "gpt2",
            "top_k": 5,
            "top_p": 0,
            "temp": 1.0, # higher => more random
            "max_len": 30,
        },
    }
}

CONFIG["yelp"] = {
    "ref0": "yelp_all_model_prediction_ref0.csv",
    "ref1": "yelp_all_model_prediction_ref1.csv",
    "label": True,
    "params": {
        "log_interval": 2000,
        "num_epochs": 100,
        "enc_lr": 1e-3,
        "dec_lr": 1.0,
        "warm_up": 10,
        "kl_start": 0.1,
        "beta1": 0.35,
        "beta2": 0.2,
        "srec_weight": 1.0,
        "reg_weight": 1.0,
        "ic_weight": 0.0,
        "aggressive": False,
        "vae_params": {
            "lstm_ni": 256,
            "lstm_nh": 1024,
            "lstm_nz": 64,
            "mlp_nz": 16,
            "dec_ni": 128,
            "dec_nh": 1024,
            "dec_dropout_in": 0.5,
            "dec_dropout_out": 0.5,
            "n_vars": 3,
        }
    }
}
CONFIG["amazon"] = {
    "ref0": "amazon_all_model_prediction_0.csv",
    "ref1": "amazon_all_model_prediction_1.csv",
    "label": True,
    "params": {
        "log_interval": 2000,
        "num_epochs": 100,
        "enc_lr": 1e-3,
        "dec_lr": 1.0,
        "warm_up": 10,
        "kl_start": 0.1,
        "beta1": 0.35,
        "beta2": 0.2,
        "srec_weight": 1.0,
        "reg_weight": 1.0,
        "ic_weight": 0.0,
        "aggressive": False,
        "vae_params": {
            "lstm_ni": 256,
            "lstm_nh": 1024,
            "lstm_nz": 64,
            "mlp_nz": 16,
            "dec_ni": 128,
            "dec_nh": 1024,
            "dec_dropout_in": 0.5,
            "dec_dropout_out": 0.5,
            "n_vars": 3,
        }
    }
}