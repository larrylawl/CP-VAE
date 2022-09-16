import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .base_network import GaussianEncoderBase

class BertForLatentConnector(GaussianEncoderBase):
    def __init__(self, syn_nz, sem_nz, device, simplex_init=nn.init.orthogonal_, name="bert-base-uncased", n_vars=3):
        super(BertForLatentConnector, self).__init__()
        self.config = AutoConfig.from_pretrained(name)
        self.enc = AutoModel.from_pretrained(name)
        self.n_vars = n_vars
        self.device = device

        # z1
        self.linear = nn.Linear(self.config.hidden_size, 2 * sem_nz, bias=False)
        self.var_embedding = nn.Parameter(torch.zeros((n_vars, sem_nz)))
        simplex_init(self.var_embedding)
        self.var_linear = nn.Linear(sem_nz, n_vars)   

        # z2 
        self.linear_syn = nn.Linear(self.config.hidden_size, 2 * syn_nz, bias=False)    

    def forward(self, inputs, attn_mask, return_p=False, to_map=True, type="sem"):
        outputs = self.enc(inputs, attn_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        if type == "syn":
           mean, logvar = self.linear_syn(cls).chunk(2, -1)
        else: # sem
            mean, logvar = self.linear(cls).chunk(2, -1)
            if to_map:
                mean, prob = self.encode_var(mean)
                if return_p: 
                    return mean, logvar, prob
        return mean, logvar

    def encode_var(self, inputs):
        # NOTE: Equation 3 in CPVAE
        logits = self.var_linear(inputs)
        prob = nn.Softmax(dim=-1)(logits)
        return torch.matmul(prob, self.var_embedding), prob
    
    # TODO: not sure why original norm is 100. shouldn't it be 1
    def orthogonal_regularizer(self, norm=1):
        # NOTE: Equation 6 in CPVAE
        tmp = torch.mm(self.var_embedding, self.var_embedding.transpose(1, 0))
        ortho_loss = torch.linalg.norm(tmp - norm * torch.diag(torch.ones(self.n_vars, device=self.device)))
        return ortho_loss

    def srec_loss(self, pos_ip, pos_attn_mask, neg_ip, neg_attn_mask):
        r, _ = self.forward(pos_ip, pos_attn_mask, to_map=False)  # mu not mapped
        pos, _ = self.forward(pos_ip, pos_attn_mask, to_map=True)  # mu
        pos_scores = (pos * r).sum(-1)
        pos_scores = pos_scores / torch.norm(r, 2, -1)
        pos_scores = pos_scores / torch.norm(pos, 2, -1)

        neg, _ = self.forward(neg_ip, neg_attn_mask, to_map=True)
        neg_scores = (neg * r).sum(-1)
        neg_scores = neg_scores / torch.norm(r, 2, -1)
        neg_scores = neg_scores / torch.norm(neg, 2, -1)

        raw_loss = torch.clamp(1 - pos_scores + neg_scores, min=0.).mean(0)  # equation 7
        srec_loss = raw_loss.mean()
        return srec_loss

    def srec_loss_mine(self, pos_ip, pos_attn_mask, neg_ip, neg_attn_mask):
        assert self.have_map
        _, _, pos_p = self.forward(pos_ip, pos_attn_mask, return_p=True)
        _, _, neg_p = self.forward(neg_ip, neg_attn_mask, return_p=True)
        srec_loss = torch.clamp(1 - pos_p - neg_p, min=0.).mean()
        return srec_loss

    def encode_syntax(self, inputs, attn_mask, nsamples=1):
        mu, logvar = self.forward(inputs, attn_mask, type="syn")
        z = self.reparameterize(mu, logvar, nsamples)
        # D[Q(z|X) || P(z)]
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
        return z, KL

    def encode_semantic(self, inputs, attn_mask, nsamples=1):
        mu, logvar = self.forward(inputs, attn_mask, type="sem")
        z = self.reparameterize(mu, logvar, nsamples)
        # D[Q(z|X) || P(z)]
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
        return z, KL