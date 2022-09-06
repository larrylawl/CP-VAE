# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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

class DecomposedVAE:
    def __init__(self, train, valid, test, bsz, save_path, logging, writer, log_interval, num_epochs,
                 enc_lr, dec_lr, warm_up, kl_start, beta1, beta2, srec_weight, reg_weight, ic_weight,
                 aggressive, text_only, vae_params):
        super(DecomposedVAE, self).__init__()
        self.bsz = bsz
        self.save_path = save_path
        self.logging = logging
        self.writer = writer
        self.log_interval = log_interval
        self.num_epochs = num_epochs
        self.enc_lr = enc_lr
        self.dec_lr = dec_lr
        self.warm_up = warm_up
        self.kl_weight = kl_start
        self.beta1 = beta1
        self.beta2 = beta2
        self.srec_weight = srec_weight
        self.reg_weight = reg_weight
        self.ic_weight = ic_weight
        self.aggressive = aggressive
        self.opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}
        self.pre_mi = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.text_only = text_only
        self.train_dl = train
        self.val_dl = valid
        self.test_dl = test

        self.vae = VAE(**vae_params)
        if self.use_cuda:
            self.vae.cuda()

        self.enc_optimizer = optim.Adam(self.vae.get_enc_params(), lr=self.enc_lr)
        self.dec_optimizer = optim.Adam(self.vae.get_dec_params(), lr=self.dec_lr)
        # self.enc_params = list(self.vae.lstm_encoder.parameters()) + \
        #     list(self.vae.mlp_encoder.parameters())
        # self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(), lr=self.dec_lr)

        self.nbatch = len(self.train_dl)
        self.anneal_rate = (1.0 - kl_start) / (warm_up * self.nbatch)

        assert not self.aggressive, "Not implemented yet."

    def train(self, epoch):
        self.vae.train()

        total_rec_loss = 0
        total_kl1_loss = 0
        total_kl2_loss = 0
        total_srec_loss = 0
        total_reg_loss = 0
        total_vae_loss = 0
        total_loss = 0

        train_dl_neg = iter(self.train_dl)
        for batch in tqdm(self.train_dl):
            
            enc_ids = batch["enc_input_ids"].to(self.device)
            enc_am = batch["enc_attention_mask"].to(self.device)
            dec_ids = batch["dec_input_ids"].to(self.device)
            rec_labels = batch["rec_labels"].to(self.device)

            batch_neg = next(train_dl_neg)
            neg_enc_ids = batch_neg["enc_input_ids"].to(self.device)
            neg_enc_am = batch_neg["enc_attention_mask"].to(self.device)

            srec_loss = self.vae.enc_sem.srec_loss(enc_ids, enc_am, neg_enc_ids, neg_enc_am)
            srec_loss = srec_loss * self.srec_weight
            reg_loss = self.vae.enc_sem.orthogonal_regularizer()
            reg_loss = reg_loss * self.reg_weight

            rec_loss, kl1_loss, kl2_loss = self.vae.loss(enc_ids, enc_am, dec_ids, rec_labels)
            kl1_loss = kl1_loss * self.beta1
            kl2_loss = kl2_loss * self.beta2
            vae_loss = rec_loss + kl1_loss + kl2_loss
            vae_loss = vae_loss.mean()

            loss = vae_loss + srec_loss + reg_loss
            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)
            self.enc_optimizer.step()
            self.dec_optimizer.step()

            total_rec_loss += rec_loss.mean().item()
            total_kl1_loss += kl1_loss.mean().item()
            total_kl2_loss += kl2_loss.mean().item()
            total_vae_loss += vae_loss.item()
            total_reg_loss += reg_loss.item()
            total_srec_loss += srec_loss.item()
            total_loss += loss.item()
            
        loss = total_loss / self.nbatch
        vae = total_vae_loss / self.nbatch
        rec = total_rec_loss / self.nbatch
        kl1 = total_kl1_loss / self.nbatch
        kl2 = total_kl2_loss / self.nbatch
        srec = total_srec_loss / self.nbatch
        reg = total_reg_loss / self.nbatch

        loss_metrics = {
            "Overall loss": loss,
            "vae": vae,
            "rec": rec,
            "kl1": kl1,
            "kl2": kl2,
            "srec": srec,
            "reg": reg,
        }
        self.logging(
                    '| train metrics of epoch {:2d} | Overall loss  {:3.2f} | vae  {:3.2f} | '
                    'recon {:3.2f} | kl1 {:3.2f} | kl2 {:3.2f} | srec {:3.2f} | reg {:3.2f}'.format(
                        epoch, loss, vae, 
                        rec, kl1, kl2, srec, reg)) 
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
            total_loss = 0

            dl = self.val_dl if split == "Val" else self.test_dl
            dl_neg = iter(dl)
            
            for batch in tqdm(dl):
                enc_ids = batch["enc_input_ids"].to(self.device)
                enc_am = batch["enc_attention_mask"].to(self.device)
                dec_ids = batch["dec_input_ids"].to(self.device)
                rec_labels = batch["rec_labels"].to(self.device)

                batch_neg = next(dl_neg)
                neg_enc_ids = batch_neg["enc_input_ids"].to(self.device)
                neg_enc_am = batch_neg["enc_attention_mask"].to(self.device)

                srec_loss = self.vae.enc_sem.srec_loss(enc_ids, enc_am, neg_enc_ids, neg_enc_am)
                srec_loss = srec_loss * self.srec_weight
                reg_loss = self.vae.enc_sem.orthogonal_regularizer()
                reg_loss = reg_loss * self.reg_weight

                rec_loss, kl1_loss, kl2_loss = self.vae.loss(enc_ids, enc_am, dec_ids, rec_labels)
                kl1_loss = kl1_loss * self.beta1
                kl2_loss = kl2_loss * self.beta2
                vae_loss = rec_loss + kl1_loss + kl2_loss
                vae_loss = vae_loss.mean()

                loss = vae_loss + srec_loss + reg_loss

                total_rec_loss += rec_loss.mean().item()
                total_kl1_loss += kl1_loss.mean().item()
                total_kl2_loss += kl2_loss.mean().item()
                total_vae_loss += vae_loss.item()
                total_reg_loss += reg_loss.item()
                total_srec_loss += srec_loss.item()
                total_loss += loss.item()
            
        loss = total_loss / self.nbatch
        vae = total_vae_loss / self.nbatch
        rec = total_rec_loss / self.nbatch
        kl1 = total_kl1_loss / self.nbatch
        kl2 = total_kl2_loss / self.nbatch
        srec = total_srec_loss / self.nbatch
        reg = total_reg_loss / self.nbatch

        loss_metrics = {
            "Overall loss": loss,
            "vae": vae,
            "rec": rec,
            "kl1": kl1,
            "kl2": kl2,
            "srec": srec,
            "reg": reg,
        }
        self.logging(
                    '| {} metrics of epoch {:2d} | Overall loss  {:3.2f} | vae  {:3.2f} | '
                    'recon {:3.2f} | kl1 {:3.2f} | kl2 {:3.2f} | srec {:3.2f} | reg {:3.2f}'.format(
                        split, epoch, loss, vae, 
                        rec, kl1, kl2, srec, reg)) 
        for k, v in loss_metrics.items():
            self.writer.add_scalar(f"{split}/{k}", v, epoch)
        return loss

    def fit(self):
        best_loss = 1e4
        decay_cnt = 0
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
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

    def save(self, path):
        self.logging("saving to %s" % path)
        model_path = os.path.join(path, "model.pt")
        torch.save(self.vae.state_dict(), model_path)

    def load(self, path):
        model_path = os.path.join(path, "model.pt")
        self.vae.load_state_dict(torch.load(model_path))


    def oldtrain(self, epoch):
        self.vae.train()

        total_rec_loss = 0
        total_kl1_loss = 0
        total_kl2_loss = 0
        total_srec_loss = 0
        total_reg_loss = 0
        total_vae_loss = 0
        total_loss = 0
        num_words = 0
        num_sents = 0

        for idx in np.random.permutation(range(self.nbatch)):
            batch_data = self.train_data[idx]
            batch_feat = self.train_feat[idx]
            sent_len, batch_size = batch_data.size()

            shift = np.random.randint(max(1, sent_len - 9))
            batch_data = batch_data[shift:min(sent_len, shift + 10), :]
            sent_len, batch_size = batch_data.size()

            target = batch_data[1:]
            num_words += (sent_len - 1) * batch_size
            num_sents += batch_size
            self.kl_weight = min(1.0, self.kl_weight + self.anneal_rate)
            beta1 = self.beta1 if self.beta1 else self.kl_weight
            beta2 = self.beta2 if self.beta2 else self.kl_weight

            loss = 0

            sub_iter = 1
            batch_data_enc = batch_data
            batch_feat_enc = batch_feat
            burn_num_words = 0
            burn_pre_loss = 1e4
            burn_cur_loss = 0
            while self.aggressive and sub_iter < 100:
                self.enc_optimizer.zero_grad()
                self.dec_optimizer.zero_grad()

                target_enc = batch_data_enc[1:]
                burn_sent_len, burn_batch_size = batch_data_enc.size()
                burn_num_words += (burn_sent_len - 1) * burn_batch_size

                logits, kl1_loss, kl2_loss, reg_ic = self.vae.loss(batch_data_enc, batch_feat_enc)
                logits = logits.view(-1, logits.size(2))
                rec_loss = F.cross_entropy(logits, target_enc.view(-1), reduction="none")
                rec_loss = rec_loss.view(-1, burn_batch_size).sum(0)
                loss = rec_loss + beta1 * kl1_loss + beta2 * kl2_loss

                burn_cur_loss = loss.sum().item()
                loss = loss.mean(dim=-1)

                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.enc_params, 0.1)
                torch.nn.utils.clip_grad_norm_(self.vae.decoder.parameters(), 5.0)

                self.enc_optimizer.step()

                id_ = np.random.randint(self.nbatch)
                batch_data_enc = self.train_data[id_]
                batch_feat_enc = self.train_feat[id_]
                burn_sent_len, burn_batch_size = batch_data_enc.size()
                shift = np.random.randint(max(1, burn_sent_len - 9))
                batch_data_enc = batch_data_enc[shift:min(burn_sent_len, shift + 10), :]

                if sub_iter % 15 == 0:
                    burn_cur_loss = burn_cur_loss / burn_num_words
                    if burn_pre_loss - burn_cur_loss < 0:
                        break
                    burn_pre_loss = burn_cur_loss
                    burn_cur_loss = burn_num_words = 0

                sub_iter += 1

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            vae_logits, vae_kl1_loss, vae_kl2_loss, reg_ic = self.vae.loss(
                batch_data, batch_feat, no_ic=self.ic_weight == 0)
            vae_logits = vae_logits.view(-1, vae_logits.size(2))
            vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
            vae_rec_loss = vae_rec_loss.view(-1, batch_size).sum(0)
            vae_loss = vae_rec_loss + beta1 * vae_kl1_loss + beta2 * vae_kl2_loss
            if self.ic_weight > 0:
                vae_loss += self.ic_weight * reg_ic
            vae_loss = vae_loss.mean()
            loss = loss + vae_loss

            total_rec_loss += vae_rec_loss.mean().item()
            total_kl1_loss += vae_kl1_loss.mean().item()
            total_kl2_loss += vae_kl2_loss.mean().item()
            total_vae_loss += vae_loss.item()

            if self.text_only:
                while True:
                    idx = np.random.choice(self.nbatch)
                    neg_feat = self.feat[idx]
                    if neg_feat.size(1) >= batch_size:
                        break
                idx = np.random.choice(batch_size, batch_size, replace=False)
                neg_feat = neg_feat[:, idx]
                var_loss, reg_loss, var_raw_loss = self.vae.var_loss(batch_feat, neg_feat, 1)
            else:
                idx = np.random.choice(self.feat.shape[1], batch_size * 10)
                neg_feat = torch.tensor(self.feat[idx], dtype=torch.float,
                                        requires_grad=False, device=self.device)
                srec_loss, reg_loss, srec_raw_loss = self.vae.var_loss(batch_feat, neg_feat, 10)
            total_srec_loss += srec_loss.item()
            total_reg_loss += reg_loss.item()
            
            loss = loss + self.srec_weight * srec_loss + self.reg_weight * reg_loss  # equation 8
            total_loss += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)
            if not self.aggressive:
                self.enc_optimizer.step()
            self.dec_optimizer.step()
        
        loss = total_loss / self.nbatch
        vae = total_vae_loss / self.nbatch
        rec = total_rec_loss / self.nbatch
        kl1 = total_kl1_loss / self.nbatch
        kl2 = total_kl2_loss / self.nbatch
        srec = total_srec_loss / self.nbatch
        reg = total_reg_loss / self.nbatch

        loss_metrics = {
            "Overall loss": loss,
            "vae": vae,
            "rec": rec,
            "kl1": kl1,
            "kl2": kl2,
            "srec": srec,
            "reg": reg,
        }
        self.logging(
                    '| train metrics of epoch {:2d} | Overall loss  {:3.2f} | vae  {:3.2f} | '
                    'recon {:3.2f} | kl1 {:3.2f} | kl2 {:3.2f} | srec {:3.2f} | reg {:3.2f}'.format(
                        epoch, loss, vae, 
                        rec, kl1, kl2, srec, reg)) 
        for k, v in loss_metrics.items():
            self.writer.add_scalar(f"Train/{k}", v, epoch)


    def oldevaluate(self, eval_data, eval_feat):
        self.vae.eval()

        beta1 = self.beta1 if self.beta1 else self.kl_weight
        beta2 = self.beta2 if self.beta2 else self.kl_weight
        total_rec_loss = 0
        total_kl1_loss = 0
        total_kl2_loss = 0
        total_srec_loss = 0
        total_reg_loss = 0
        total_vae_loss = 0
        total_loss = 0
        total_mi1 = 0
        total_mi2 = 0
        num_sents = 0
        num_words = 0
        nbatch_eval = len(eval_data)

        with torch.no_grad():
            for batch_data, batch_feat in zip(eval_data, eval_feat):
                sent_len, batch_size = batch_data.size()
                shift = np.random.randint(max(1, sent_len - 9))
                batch_data = batch_data[shift:min(sent_len, shift + 10), :]
                sent_len, batch_size = batch_data.size()
                target = batch_data[1:]

                num_sents += batch_size
                num_words += (sent_len - 1) * batch_size

                vae_logits, vae_kl1_loss, vae_kl2_loss, _ = self.vae.loss(
                    batch_data, batch_feat)
                vae_logits = vae_logits.view(-1, vae_logits.size(2))
                vae_rec_loss = F.cross_entropy(vae_logits, target.view(-1), reduction="none")
                vae_rec_loss = vae_rec_loss.view(-1, batch_size).sum(0)
                vae_loss = vae_rec_loss + beta1 * vae_kl1_loss + beta2 * vae_kl2_loss
                vae_loss = vae_loss.mean()

                total_rec_loss += vae_rec_loss.mean().item()
                total_kl1_loss += vae_kl1_loss.mean().item()
                total_kl2_loss += vae_kl2_loss.mean().item()
                total_vae_loss += vae_loss.item()

                mi1, mi2 = self.vae.calc_mi_q(batch_data, batch_feat)
                total_mi1 += mi1 * batch_size
                total_mi2 += mi2 * batch_size

                if self.text_only:
                    raise NotImplementedError
                else:
                    idx = np.random.choice(self.feat.shape[1], batch_size * 10)
                    neg_feat = torch.tensor(self.feat[idx], dtype=torch.float,
                                            requires_grad=False, device=self.device)
                    srec_loss, reg_loss, srec_raw_loss = self.vae.var_loss(batch_feat, neg_feat, 10)
                total_srec_loss += srec_loss.item()
                total_reg_loss += reg_loss.item()

                loss = vae_loss + self.srec_weight * srec_loss + self.reg_weight * reg_loss  # equation 8
                total_loss += loss.item()

        cur_rec_loss = total_rec_loss / nbatch_eval
        cur_kl1_loss = total_kl1_loss / nbatch_eval
        cur_kl2_loss = total_kl2_loss / nbatch_eval
        cur_vae_loss = total_vae_loss / nbatch_eval
        cur_mi1 = total_mi1 / num_sents  
        cur_mi2 = total_mi2 / num_sents
        cur_srec_loss = total_srec_loss / nbatch_eval
        cur_reg_loss = total_reg_loss / nbatch_eval
        cur_loss = total_loss / nbatch_eval
        
        return cur_loss, cur_vae_loss, cur_rec_loss, cur_kl1_loss, cur_kl2_loss, cur_mi1, cur_mi2, cur_srec_loss, cur_reg_loss
