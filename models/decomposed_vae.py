# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import umap
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .vae import DecomposedVAE as VAE
import numpy as np
import time
import os
from tqdm import tqdm
from copy import copy
from .utils import dropout_for_longtensor
import matplotlib.pyplot as plt

class DecomposedVAE:
    def __init__(self, train, valid, test, bsz, save_path, logging, to_plot, writer, log_interval, num_epochs,
                 enc_lr, dec_lr, warm_up, kl_start, beta1, beta2, cycles, proportion, srec_weight, reg_weight, style_weight, ic_weight,
                 aggressive, text_only, vae_params, debug, accum_iter):
        super(DecomposedVAE, self).__init__()
        self.bsz = bsz
        self.save_path = save_path
        self.logging = logging
        self.writer = writer
        self.to_plot = to_plot
        self.log_interval = log_interval
        self.num_epochs = num_epochs
        self.enc_lr = enc_lr
        self.dec_lr = dec_lr
        self.warm_up = warm_up
        self.kl_weight = kl_start
        self.accum_iter = accum_iter
        self.cycles = cycles
        self.proportion = proportion
        self.srec_weight = srec_weight
        self.reg_weight = reg_weight
        self.style_weight = style_weight
        self.ic_weight = ic_weight
        self.aggressive = aggressive
        self.opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}
        self.pre_mi = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        self.save_simplex_path = os.path.join(self.save_path, "simplex")
        self.save_umap_path = os.path.join(self.save_path, "umap")

        self.text_only = text_only
        self.train_dl = train
        self.val_dl = valid
        self.test_dl = test

        self.vae = VAE(**vae_params)
        if self.use_cuda:
            self.vae.cuda()

        self.enc_optimizer = optim.Adam(self.vae.get_enc_params(), lr=self.enc_lr)
        self.dec_optimizer = optim.Adam(self.vae.get_dec_params(), lr=self.dec_lr)

        self.nbatch = len(self.train_dl)
        self.anneal_rate = (1.0 - kl_start) / (warm_up * self.nbatch)

        assert not self.aggressive, "Not implemented yet."

    def train(self, epoch):
        self.vae.train()
        # kl_weight = self.cyclic_annealing(epoch)
        # keep_prob = 1 - (kl_weight / 2)
        self.kl_weight = min(1.0, self.kl_weight + self.anneal_rate)

        total_rec_loss = 0
        total_kl1_loss = 0
        total_kl2_loss = 0
        total_srec_loss = 0
        total_reg_loss = 0
        total_vae_loss = 0
        total_style_loss = 0
        total_loss = 0

        train_dl_neg = iter(self.train_dl)

        for batch_idx, (batch, buddy_batch) in tqdm(enumerate(self.train_dl), total=len(self.train_dl)):
            
            enc_ids = batch["enc_input_ids"].to(self.device)
            enc_am = batch["enc_attention_mask"].to(self.device)
            # enc_am = dropout_for_longtensor(enc_am, keep_prob=keep_prob)  # dropout as noise
            dec_ids = batch["dec_input_ids"].to(self.device)
            rec_labels = batch["rec_labels"].to(self.device)
            style_labels = batch["labels"].to(self.device)

            ### TextSettr idea ###
            # bd_enc_ids = buddy_batch["enc_input_ids"].to(self.device)
            # bd_enc_am = buddy_batch["enc_attention_mask"].to(self.device)

            # neg_batch, _ = next(train_dl_neg)
            # neg_enc_ids = neg_batch["enc_input_ids"].to(self.device)
            # neg_enc_am = neg_batch["enc_attention_mask"].to(self.device)

            ### srec loss ###
            # srec_loss = self.vae.enc.srec_loss(enc_ids, enc_am, neg_enc_ids, neg_enc_am)
            # srec_loss = srec_loss * self.srec_weight
            srec_loss = torch.cuda.FloatTensor(1).fill_(0)
            reg_loss = self.vae.enc.orthogonal_regularizer()
            reg_loss = reg_loss * self.reg_weight

            rec_loss, kl1_loss, kl2_loss, _, style_loss, _ = self.vae.loss(enc_ids, enc_am, None, None, dec_ids, rec_labels, style_labels)
            style_loss = style_loss * self.style_weight
            kl1_loss = kl1_loss * self.kl_weight
            kl2_loss = kl2_loss * self.kl_weight
            vae_loss = rec_loss + kl1_loss + kl2_loss
            vae_loss = vae_loss.mean()

            # print(f"vae_loss: {vae_loss}")
            # print(f"srec_loss: {srec_loss}")
            # print(f"reg_loss: {reg_loss}")
            # print(f"style_loss: {style_loss}")
            loss = vae_loss + srec_loss + reg_loss + style_loss
            norm_loss = loss / self.accum_iter  # gradient accumulation
            norm_loss.backward()

            if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(self.train_dl)):
                nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)
                self.enc_optimizer.step()
                self.dec_optimizer.step()
                self.enc_optimizer.zero_grad()
                self.dec_optimizer.zero_grad()

            total_rec_loss += rec_loss.mean().item()
            total_kl1_loss += kl1_loss.mean().item()
            total_kl2_loss += kl2_loss.mean().item()
            total_vae_loss += vae_loss.item()
            total_reg_loss += reg_loss.item()
            total_srec_loss += srec_loss.item()
            total_style_loss += style_loss.item()
            total_loss += loss.item()
            
            if self.debug:
                break
            
        loss = total_loss / self.nbatch
        vae = total_vae_loss / self.nbatch
        rec = total_rec_loss / self.nbatch
        kl1 = total_kl1_loss / self.nbatch
        kl2 = total_kl2_loss / self.nbatch
        srec = total_srec_loss / self.nbatch
        reg = total_reg_loss / self.nbatch
        style = total_style_loss / self.nbatch

        loss_metrics = {
            "Overall loss": loss,
            "vae": vae,
            "rec": rec,
            "kl1": kl1,
            "kl2": kl2,
            "srec": srec,
            "reg": reg,
            "style": style,
        }
        self.logging(
                    '| train metrics of epoch {:2d} | Overall loss  {:3.2f} | vae  {:3.2f} | '
                    'recon {:3.2f} | kl1 {:3.2f} | kl2 {:3.2f} | srec {:3.2f} | reg {:3.2f} | style {:3.2f}'.format(
                        epoch, loss, vae, 
                        rec, kl1, kl2, srec, reg, style)) 
        for k, v in loss_metrics.items():
            self.writer.add_scalar(f"Train/{k}", v, epoch)


    def evaluate(self, split="Val", epoch=0):
        self.vae.eval()

        with torch.no_grad():
            total_rec_loss = 0
            total_kl1_loss = 0
            total_kl2_loss = 0
            total_srec_loss = 0
            total_reg_loss = 0
            total_vae_loss = 0
            total_style_loss = 0
            total_loss = 0

            z1_accum = []  # for umap
            to_plot_in_this_epoch = self.to_plot % 3 == 1  # plot every 3 epochs

            if to_plot_in_this_epoch:
                p_accum = []  # for visualisation over epoch
                labels_accum = []
                if not os.path.exists(os.path.join(self.save_simplex_path)):
                    os.makedirs(self.save_simplex_path)
                    os.makedirs(self.save_umap_path)

            dl = self.val_dl if split == "Val" else self.test_dl
            dl_neg = iter(dl)
            nbatch = len(dl)
            
            for batch_idx, (batch, buddy_batch) in tqdm(enumerate(dl), total=len(dl)):
                enc_ids = batch["enc_input_ids"].to(self.device)
                enc_am = batch["enc_attention_mask"].to(self.device)
                dec_ids = batch["dec_input_ids"].to(self.device)
                rec_labels = batch["rec_labels"].to(self.device)
                style_labels = batch["labels"].to(self.device)

                # bd_enc_ids = buddy_batch["enc_input_ids"].to(self.device)
                # bd_enc_am = buddy_batch["enc_attention_mask"].to(self.device)

                # neg_batch, _ = next(dl_neg)
                # neg_enc_ids = neg_batch["enc_input_ids"].to(self.device)
                # neg_enc_am = neg_batch["enc_attention_mask"].to(self.device)

                # srec_loss = self.vae.enc_sem.srec_loss(enc_ids, enc_am, neg_enc_ids, neg_enc_am)
                # srec_loss = self.vae.enc.srec_loss(enc_ids, enc_am, neg_enc_ids, neg_enc_am)
                # srec_loss = srec_loss * self.srec_weight
                srec_loss = torch.cuda.FloatTensor(1).fill_(0)
                reg_loss = self.vae.enc.orthogonal_regularizer()
                reg_loss = reg_loss * self.reg_weight

                rec_loss, kl1_loss, kl2_loss, p, style_loss, z1 = self.vae.loss(enc_ids, enc_am, None, None, dec_ids, rec_labels, style_labels)
                style_loss = style_loss * self.style_weight
                kl1_loss = kl1_loss * self.kl_weight
                kl2_loss = kl2_loss * self.kl_weight
                vae_loss = rec_loss + kl1_loss + kl2_loss
                vae_loss = vae_loss.mean()

                loss = vae_loss + srec_loss + reg_loss + style_loss

                total_rec_loss += rec_loss.mean().item()
                total_kl1_loss += kl1_loss.mean().item()
                total_kl2_loss += kl2_loss.mean().item()
                total_vae_loss += vae_loss.item()
                total_reg_loss += reg_loss.item()
                total_srec_loss += srec_loss.item()
                total_style_loss += style_loss.item()
                total_loss += loss.item()

                if to_plot_in_this_epoch:
                    p_accum.append(p.cpu().detach().numpy())
                    labels_accum.extend(batch["labels"].tolist())
                    z1_accum.append(z1.cpu().detach().numpy())

                if self.debug:
                    break
            
        loss = total_loss / nbatch
        vae = total_vae_loss / nbatch
        rec = total_rec_loss / nbatch
        kl1 = total_kl1_loss / nbatch
        kl2 = total_kl2_loss / nbatch
        srec = total_srec_loss / nbatch
        reg = total_reg_loss / nbatch
        style = total_style_loss / nbatch

        loss_metrics = {
            "Overall loss": loss,
            "vae": vae,
            "rec": rec,
            "kl1": kl1,
            "kl2": kl2,
            "srec": srec,
            "reg": reg,
            "style": style,
        }
        self.logging(
                    '| {} metrics of epoch {:2d} | Overall loss  {:3.2f} | vae  {:3.2f} | '
                    'recon {:3.2f} | kl1 {:3.2f} | kl2 {:3.2f} | srec {:3.2f} | reg {:3.2f} | style {:3.2f}'.format(
                        split, epoch, loss, vae, 
                        rec, kl1, kl2, srec, reg, style)) 
        for k, v in loss_metrics.items():
            self.writer.add_scalar(f"{split}/{k}", v, epoch)
        
        if to_plot_in_this_epoch:
            # plotting simplex p
            p = np.concatenate(p_accum)
            labels = labels_accum
            self.plot_simplex(p, labels, epoch)

            # plotting umap of z1
            z1 = np.concatenate(z1_accum)
            self.plot_umap(z1, labels, epoch)

        tracked_loss = rec + style
        return tracked_loss  # return rec loss as kl losses linearly increases with epochs

    def fit(self):
        best_loss = 1e4
        decay_cnt = 0
        for epoch in range(1, self.num_epochs + 1):
            self.train(epoch)
            val_loss = self.evaluate(epoch=epoch, split="Val")

            if self.aggressive:
                cur_mi = val_loss[5]
                self.logging("pre mi: %.4f, cur mi:%.4f" % (self.pre_mi, cur_mi))
                if cur_mi < self.pre_mi:
                    self.aggressive = False
                    self.logging("STOP BURNING")

                self.pre_mi = cur_mi

            if val_loss < best_loss:
                self.save(self.save_path)
                best_loss = val_loss

            if val_loss > self.opt_dict["best_loss"]:
                self.opt_dict["not_improved"] += 1
                if self.opt_dict["not_improved"] >= 2 and epoch >= 5:
                    self.opt_dict["not_improved"] = 0
                    self.opt_dict["lr"] = self.opt_dict["lr"] * 0.5
                    self.load(self.save_path)
                    decay_cnt += 1
                    self.dec_optimizer = optim.Adam(self.vae.get_dec_params(), lr=self.opt_dict["lr"])
            else:
                self.opt_dict["not_improved"] = 0
                self.opt_dict["best_loss"] = val_loss

            if decay_cnt == 5:
                break

        return best_loss

    def cyclic_annealing(self, epoch):
        # Equation 6 in paper: Cyclical Annealing Schedule
        t = epoch
        epochs_per_cycle = self.num_epochs / self.cycles
        tau = ((t - 1) % math.ceil(epochs_per_cycle)) / (epochs_per_cycle)
        if tau <= self.proportion:
            return tau / self.proportion  # linear annealing
        else:
            return 1       

    def save(self, path):
        self.logging("saving to %s" % path)
        model_path = os.path.join(path, "model.pt")
        torch.save(self.vae.state_dict(), model_path)

    def load(self, path):
        model_path = os.path.join(path, "model.pt")
        self.vae.load_state_dict(torch.load(model_path))

    def plot_simplex(self, p, labels, epoch):
        plt.clf()
        fp = os.path.join(self.save_simplex_path, f"{epoch}.jpg")
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        ax.scatter(p[:,0], p[:,1], c=labels)
        plt.title(f"Scatter plot of simplex for epoch {epoch}")
        plt.savefig(fp)
        self.logging(f"saving simplex plot to {fp}")
    
    def plot_umap(self, z1, labels, epoch):
        plt.clf()
        fp = os.path.join(self.save_umap_path, f"{epoch}.jpg")
        reducer = umap.UMAP()
        z1 = reducer.fit_transform(z1)
        plt.clf()
        plt.scatter(z1[:, 0], z1[:, 1], c=labels, cmap='Spectral', s=5)
        plt.gca().set_aspect('auto', 'datalim')
        plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
        plt.title(f'UMAP projection for epoch {epoch}', fontsize=24)
        plt.savefig(fp)
        self.logging(f"saving umap plot to {fp}")