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


import torch
import torch.nn as nn

from .utils import uniform_initializer, value_initializer, gumbel_softmax
from .base_network import LSTMEncoder, LSTMDecoder, SemMLPEncoder, SemLSTMEncoder
from itertools import chain
from .pytorch_transformers.modeling_gpt2 import GPT2ForLatentConnector, GPT2Config
from models.bert_enc import BertForLatentConnector
# from .pytorch_transformers.modeling_bert import BertForLatentConnector, BertConfig

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
    def __init__(self, enc_name, dec_name, syn_nz, sem_nz, n_vars, device, top_k, top_p, temp, max_len):
        super(DecomposedVAE, self).__init__()
        self.enc = BertForLatentConnector(syn_nz=syn_nz, sem_nz=sem_nz, device=device, n_vars=n_vars, name=enc_name)

        # self.enc_syn = BertForLatentConnector(syn_nz, enc_name)
        # simplex_init = nn.init.orthogonal_
        # self.enc_sem = SemMLPEncoder(ni=ni, nz=sem_nz, n_vars=n_vars, simplex_init=simplex_init, device=device)
        # self.enc_sem = BertForLatentConnector(sem_nz, device=device, name=enc_name, have_map=True, n_vars=n_vars, simplex_init=simplex_init)
        self.dec_nz = sem_nz + syn_nz
        self.dec_config = GPT2Config.from_pretrained(dec_name)
        setattr(self.dec_config, "latent_size", self.dec_nz)
        self.dec = GPT2ForLatentConnector.from_pretrained(dec_name, config=self.dec_config)
        self.device = device
        self.top_k = top_k
        self.top_p = top_p
        self.temp = temp
        self.max_len = max_len
        self.style_loss = nn.CrossEntropyLoss()
        
        ### Freezing shared encoder parameters ###
        # self.enc.freeze_shared_encoder_params()
        

    def loss(self, enc_ids, enc_attn_mask, bd_enc_ids, bd_enc_attn_mask, dec_ids, rec_labels, style_labels, nsamples=1):
        # z1, KL1 = self.enc.encode_semantic(bd_enc_ids, bd_enc_attn_mask, nsamples=nsamples)
        z1, KL1, logits = self.enc.encode_semantic(enc_ids, enc_attn_mask, style_labels, nsamples=nsamples)  # z1 size: (bsz, sem_nz)
        z2, KL2 = self.enc.encode_syntax(enc_ids, enc_attn_mask, nsamples=nsamples)
        z1 = z1.squeeze()
        z2 = z2.squeeze()
        z = torch.cat([z1, z2], -1)
        op = self.dec(input_ids=dec_ids, past=z, labels=rec_labels, label_ignore=-100)
        rec_loss = op[0]
        style_loss = self.style_loss(logits, style_labels)
        return rec_loss, KL1, KL2, logits, style_loss, z1
        
    # def encode_syntax(self, x, enc_attn_mask, nsamples=1):
    #     return self.enc_syn.encode(x, enc_attn_mask, nsamples)

    # def encode_semantic(self, x, enc_attn_mask, nsamples=1):
    #     return self.enc_sem.encode(x, enc_attn_mask, nsamples)

    def get_enc_params(self):
        return self.enc.parameters()
        # return chain(self.enc_syn.parameters(), self.enc_sem.parameters())

    def get_dec_params(self):
        return self.dec.parameters()

    def get_bos_token_id_tensor(self, bsz):
        op = self.dec_config.bos_token_id
        op = torch.tensor(op, dtype=torch.long, device=self.device)
        op = op.unsqueeze(0).repeat(bsz, 1)
        return op

    def sample_sequence_conditional_batch(self, context, past):
        # context: a single id of <BOS>
        # past: (B, past_seq_len dim_h)
        generated = context
        # context = self.dec_config.bos_token_id
        # num_samples = past.size(0)
        # context = torch.tensor(context, dtype=torch.long, device=past.device)
        # context = context.unsqueeze(0).repeat(num_samples, 1)
        # generated = context # (B, 1)

        # with torch.no_grad():
        while generated.size(-1) < self.max_len:
            inputs = {'input_ids': generated, 'past': past}
            outputs = self.dec(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            lm_logits = outputs[0]

            # softmax sample
            next_tokens_logits = lm_logits[:, -1, :] / self.temp  # (B, 1, vocab_size)
            filtered_logits = self.top_k_top_p_filtering_batch(next_tokens_logits, top_k=self.top_k, top_p=self.top_p)  # (B, 1, vocab_size)
            filtered_logits = nn.Softmax(dim=-1)(filtered_logits)
            next_tokens = torch.multinomial(filtered_logits, num_samples=1)   # (B, 1)
            generated = torch.cat((generated, next_tokens), dim=1)  # (B, seq_len+1)

            not_finished = next_tokens != self.dec_config.eos_token_id
            if torch.sum(not_finished) == 0:
                break

        return generated    # (B, seq_len)

    def top_k_top_p_filtering_batch(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

        top_k = min(top_k, logits.size(-1))  # Safety check

        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            threshold = torch.topk(logits, top_k, dim=-1)[0][:, -1, None]
            logits.masked_fill_(logits < threshold, filter_value)   #  (B, vocab_size)

        if top_p > 0.0:
            raise NotImplementedError("Strange bug.")
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)         # (B, vocab_size)
            cumulative_probs = torch.cumsum(nn.Softmax(dim=-1)(sorted_logits), dim=-1)   # (B, vocab_size)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p  # (B, vocab_size)

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]

            logits.masked_fill_(logits[indices_to_remove], filter_value)

        return logits 
