
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
import math
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import pickle
import torch
from torch.utils.data import Dataset
# ---------------- Hugging-Face imports ----------------
from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertModel, BertLayer, BertSelfAttention, BertOnlyMLMHead
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# ---------------- Gaussian kernel ---------------------
class GaussianParameters(nn.Module):
    def __init__(self, K=10):
        super().__init__()
        # substrate kernels
        self.mu_sub  = nn.Parameter(torch.randn(K))
        self.sig_sub = nn.Parameter(torch.ones(K))
        self.b_sub   = nn.Parameter(torch.zeros(K))
        # projection layers
        self.w1 = nn.Linear(K, K)
        self.w2 = nn.Linear(K, 1)

def gelu(x):
    return x * 0.5 * (1. + torch.erf(x / math.sqrt(2.0)))

# ---------------- Self-Attention with bias -------------
class DistanceBertSelfAttention(BertSelfAttention):
    def __init__(self, cfg, gp: GaussianParameters):
        super().__init__(cfg)
        self.gp = gp

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None,
        output_attentions=False, distance_matrix=None
    ):
        # standard projections
        q, k, v = (self.transpose_for_scores(p) for p in
                   (self.query(hidden_states),
                    self.key(hidden_states),
                    self.value(hidden_states)))
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores /= math.sqrt(self.attention_head_size)

        # add Gaussian distance bias
        if distance_matrix is not None:
            mu, sig, b = self.gp.mu_sub, self.gp.sig_sub, self.gp.b_sub
            # expand and compute RBF
            dist = distance_matrix.unsqueeze(1).unsqueeze(-1) + b.view(1,1,1,-1)
            psi  = torch.exp(-0.5 * ((dist - mu)/sig)**2) / (math.sqrt(2*math.pi)*sig)
            psi  = gelu(self.gp.w1(psi))
            phi  = self.gp.w2(psi).squeeze(-1)

            scores = scores + phi.expand_as(scores)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_probs = nn.Softmax(dim=-1)(scores)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        context = torch.matmul(attn_probs, v)

        # merge heads
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), self.all_head_size)

        return (context, attn_probs) if output_attentions else (context,)


class DistanceBertLayer(BertLayer):
    def __init__(self, cfg, gp):
        super().__init__(cfg)
        self.attention.self = DistanceBertSelfAttention(cfg, gp)

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None,
        output_attentions=False, distance_matrix=None
    ):
        att_out = self.attention.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask[0] if head_mask is not None else None,
            output_attentions=output_attentions,
            distance_matrix=distance_matrix,
        )
        context = att_out[0]
        attn    = att_out[1] if output_attentions else None

        hidden_states = self.attention.output(context, hidden_states)
        inter = self.intermediate(hidden_states)
        layer_out = self.output(inter, hidden_states)

        return (layer_out, attn) if output_attentions else (layer_out,)


class DistanceBertModel(BertModel):
    def __init__(self, cfg, gp):
        super().__init__(cfg)
        # replace encoder layers
        self.encoder.layer = nn.ModuleList(
            [DistanceBertLayer(cfg, gp) for _ in range(cfg.num_hidden_layers)]
        )
        self.init_weights()

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None,
        output_attentions=False, output_hidden_states=False, return_dict=True,
        distance_matrix=None
    ):
        emb = self.embeddings(
            input_ids=input_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        B, L = emb.size()[:2]
        if attention_mask is None:
            attention_mask = torch.ones((B, L), device=emb.device)
        ext_mask = self.get_extended_attention_mask(attention_mask, (B, L), emb.device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        all_hid = () if output_hidden_states else None
        all_att = () if output_attentions else None
        h = emb
        for i, layer in enumerate(self.encoder.layer):
            if output_hidden_states:
                all_hid += (h,)
            out_tuple = layer(
                h, attention_mask=ext_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                output_attentions=output_attentions,
                distance_matrix=distance_matrix,
            )
            h = out_tuple[0]
            if output_attentions:
                all_att += (out_tuple[1],)
        if output_hidden_states:
            all_hid += (h,)

        pool = self.pooler(h) if self.pooler is not None else None
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=h,
            pooler_output=pool,
            hidden_states=all_hid,
            attentions=all_att
        )

# ---------------- MLM Head -----------------------------
class DistanceBertForMLM(BertPreTrainedModel):
    def __init__(self, cfg, gp):
        super().__init__(cfg)
        self.bert = DistanceBertModel(cfg, gp)
        self.cls  = BertOnlyMLMHead(cfg)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None,
                labels=None, distance_matrix=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            distance_matrix=distance_matrix,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        logits = self.cls(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1), ignore_index=-100
            )
        return out.last_hidden_state,{"loss": loss, "logits": logits}


