# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2020-present, Juxian He
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Code is based on the VAE lagging encoder (https://arxiv.org/abs/1901.05534) implementation
# from https://github.com/jxhe/vae-lagging-encoder by Junxian He
#################################################################################################


from models.bert_enc import BertForLatentConnector
import torch
import torch.nn as nn

from .utils import uniform_initializer, value_initializer, gumbel_softmax
from .base_network import LSTMEncoder, LSTMDecoder, SemMLPEncoder, SemLSTMEncoder
from itertools import chain
from .pytorch_transformers.modeling_gpt2 import GPT2ForLatentConnector, GPT2Config

class VAE(nn.Module):
    def __init__(self, ni, nz, enc_nh, dec_nh, dec_dropout_in, dec_dropout_out, vocab, device):
        super(VAE, self).__init__()
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)
        self.encoder = LSTMEncoder(ni, enc_nh, nz, len(vocab), model_init, enc_embed_init)
        self.decoder = LSTMDecoder(
            ni, dec_nh, nz, dec_dropout_in, dec_dropout_out, vocab,
            model_init, dec_embed_init, device)

    def cuda(self):
        self.encoder.cuda()
        self.decoder.cuda()

    def encode(self, x, nsamples=1):
        return self.encoder.encode(x, nsamples)

    def decode(self, x, z):
        return self.decoder(x, z)

    def loss(self, x, nsamples=1):
        z, KL = self.encode(x, nsamples)
        outputs = self.decode(x[:-1], z)
        return outputs, KL

    def calc_mi_q(self, x):
        return self.encoder.calc_mi(x)

class DecomposedVAE(nn.Module):
    def __init__(self, enc_name, dec_name, syn_nz, sem_nz, n_vars, device):
        super(DecomposedVAE, self).__init__()
        self.enc_syn = BertForLatentConnector(syn_nz, enc_name)
        self.enc_sem = BertForLatentConnector(sem_nz, device=device, name=enc_name, have_map=True, n_vars=n_vars)
        self.dec_nz = syn_nz + sem_nz
        decoder_config = GPT2Config.from_pretrained(dec_name)
        setattr(decoder_config, "latent_size", self.dec_nz)
        self.dec = GPT2ForLatentConnector.from_pretrained(dec_name, config=decoder_config, latent_size=self.dec_nz, latent_as_gpt_emb=False)
        self.device = device

    def loss(self, enc_ids, enc_attn_mask, dec_ids, rec_labels, nsamples=1):
        z1, KL1 = self.encode_semantic(enc_ids, enc_attn_mask, nsamples)
        z2, KL2 = self.encode_syntax(enc_ids, enc_attn_mask, nsamples)
        z = torch.cat([z1, z2], -1).squeeze()
        op = self.dec(input_ids=dec_ids, past=z, labels=rec_labels, label_ignore=-100)
        rec_loss = op[0]
        return rec_loss, KL1, KL2
        
    def encode_syntax(self, x, enc_attn_mask, nsamples=1):
        return self.enc_syn.encode(x, enc_attn_mask, nsamples)

    def encode_semantic(self, x, enc_attn_mask, nsamples=1):
        return self.enc_sem.encode(x, enc_attn_mask, nsamples)

    def get_enc_params(self):
        return chain(self.enc_syn.parameters(), self.enc_sem.parameters())

    def get_dec_params(self):
        return self.dec.parameters()


class OldDecomposedVAE(nn.Module):
    def __init__(self, lstm_ni, lstm_nh, lstm_nz, mlp_ni, mlp_nz,
                 dec_ni, dec_nh, dec_dropout_in, dec_dropout_out,
                 vocab, n_vars, device, text_only):
        super(OldDecomposedVAE, self).__init__()
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)
        
        self.lstm_encoder = LSTMEncoder(
            lstm_ni, lstm_nh, lstm_nz, len(vocab), model_init, enc_embed_init)
        if text_only:
            self.mlp_encoder = SemLSTMEncoder(
                lstm_ni, lstm_nh, mlp_nz, len(vocab), n_vars, model_init, enc_embed_init, device)
        else:
            self.mlp_encoder = SemMLPEncoder(
                mlp_ni, mlp_nz, n_vars, model_init, device)
        self.decoder = LSTMDecoder(
            dec_ni, dec_nh, lstm_nz + mlp_nz, dec_dropout_in, dec_dropout_out, vocab,
            model_init, dec_embed_init, device)

    def encode_syntax(self, x, nsamples=1):
        return self.lstm_encoder.encode(x, nsamples)

    def encode_semantic(self, x, nsamples=1):
        return self.mlp_encoder.encode(x, nsamples)

    def decode(self, x, z):
        return self.decoder(x, z)

    def var_loss(self, pos, neg, neg_samples):
        r, _ = self.mlp_encoder(pos, True)  # mu not mapped
        pos = self.mlp_encoder.encode_var(r)  # mu
        pos_scores = (pos * r).sum(-1)
        pos_scores = pos_scores / torch.norm(r, 2, -1)
        pos_scores = pos_scores / torch.norm(pos, 2, -1)
        neg, _ = self.mlp_encoder(neg)
        neg_scores = (neg * r.repeat(neg_samples, 1)).sum(-1)
        neg_scores = neg_scores / torch.norm(r.repeat(neg_samples, 1), 2, -1)
        neg_scores = neg_scores / torch.norm(neg, 2, -1)
        neg_scores = neg_scores.view(neg_samples, -1)
        pos_scores = pos_scores.unsqueeze(0).repeat(neg_samples, 1)
        raw_loss = torch.clamp(1 - pos_scores + neg_scores, min=0.).mean(0)  # equation 7
        srec_loss = raw_loss.mean()
        reg_loss = self.mlp_encoder.orthogonal_regularizer()
        return srec_loss, reg_loss, raw_loss.sum()

    def get_var_prob(self, inputs):
        _, p = self.mlp_encoder.encode_var(inputs, True)
        return p

    def loss(self, x, feat, tau=1.0, nsamples=1, no_ic=True):
        z1, KL1 = self.encode_syntax(x, nsamples)
        z2, KL2 = self.encode_semantic(feat, nsamples)
        z = torch.cat([z1, z2], -1)
        outputs = self.decode(x[:-1], z)
        if no_ic:
            reg_ic = torch.zeros(10)
        else:
            soft_outputs = gumbel_softmax(outputs, tau)
            log_density = self.lstm_encoder.eval_inference_dist(soft_outputs, z1)
            logit = log_density.exp()
            reg_ic = -torch.log(torch.sigmoid(logit))
        return outputs, KL1, KL2, reg_ic

    def calc_mi_q(self, x, feat):
        # TODO: no idea whats mi1, mi2
        mi1 = self.lstm_encoder.calc_mi(x)
        mi2 = self.mlp_encoder.calc_mi(feat)
        return mi1, mi2
