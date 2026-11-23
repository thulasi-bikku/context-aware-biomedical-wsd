"""Phase 3 and fusion (Eqs. 16-28) in PyTorch."""
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    def __init__(self, n_buckets=100_000, dim=200, l_max=2, dropout=0.3, use_mention=False):
        super().__init__()
        self.emb = nn.EmbeddingBag(n_buckets, dim, mode="mean")
        nn.init.normal_(self.emb.weight, std=0.1)
        self.q = nn.Parameter(torch.randn(dim) * 0.1)               # attention query (Eq. 17)
        self.lam_logits = nn.Parameter(torch.zeros(l_max + 1))     # softmax -> lambda_l (Eq. 17 mixture)
        self.beta = nn.Parameter(torch.ones(4))                    # beta_1..beta_4 (Eq. 21)
        self.drop = nn.Dropout(dropout)
        self.use_mention = use_mention
        self.dim = dim

    def bag(self, id_lists):
        flat = torch.tensor([i for l in id_lists for i in l], dtype=torch.long)
        offs = torch.tensor([0] + list(itertools.accumulate(len(l) for l in id_lists))[:-1], dtype=torch.long)
        return self.emb(flat, offs)

    def forward(self, ex):
        """Return (logits (C,), p_ctx (C, V)) for one example."""
        tok = self.bag(ex.ctx_ids)                                # (n, dim)
        a = torch.softmax(tok @ self.q, dim=0)                    # attention weights
        h = (a[:, None] * tok).sum(0)                             # h_x
        if self.use_mention:
            h = h + self.bag([ex.mention_ids])[0]
        h = self.drop(h)
        vc = torch.stack([self.bag(t).mean(0) if len(t) else torch.zeros(self.dim) for t in ex.cand_term_ids])
        s_emb = F.cosine_similarity(h[None, :], vc, dim=1, eps=1e-8)
        lam = torch.softmax(self.lam_logits, 0)
        layers = torch.as_tensor(ex.layers, dtype=torch.float32)  # (C, L+1, V)
        p_ctx = torch.einsum("l,clv->cv", lam, layers)
        p_d = torch.as_tensor(ex.p_d, dtype=torch.float32)
        s_prob = (p_d[None, :] * torch.log(p_ctx.clamp_min(1e-12))).sum(1)   # -H(P_d, P_ctx)
        cov = torch.as_tensor(ex.cov, dtype=torch.float32)
        lp = torch.as_tensor(ex.logprior, dtype=torch.float32)
        logits = self.beta[0] * s_prob + self.beta[1] * s_emb + self.beta[2] * cov + self.beta[3] * lp
        return logits, p_ctx

    def loss(self, ex, gamma=0.1):
        logits, p_ctx = self(ex)
        l_sup = F.cross_entropy(logits[None, :], torch.tensor([ex.gold]))
        p_d = torch.as_tensor(ex.p_d, dtype=torch.float32)
        l_ctx = (p_d * (torch.log(p_d.clamp_min(1e-12)) - torch.log(p_ctx[ex.gold].clamp_min(1e-12)))).sum()
        return l_sup + gamma * l_ctx

    @torch.no_grad()
    def predict_proba(self, ex):
        self.eval()
        return torch.softmax(self(ex)[0], 0).numpy()


def train(model, train_ex, val_ex=None, epochs=10, lr=1e-3, batch_size=32, gamma=0.1, rho=1e-6, patience=3, seed=0):
    """Adam training with the joint objective of Eq. 28 and early stopping on validation accuracy."""
    random.seed(seed); torch.manual_seed(seed)
    train_ex = [e for e in train_ex if e.gold >= 0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        random.shuffle(train_ex)
        for i in range(0, len(train_ex), batch_size):
            batch = train_ex[i:i + batch_size]
            loss = sum(model.loss(e, gamma) for e in batch) / len(batch)
            loss = loss + 0.5 * rho * sum((p ** 2).sum() for p in model.parameters())
            opt.zero_grad(); loss.backward(); opt.step()
        if val_ex:
            acc = np.mean([int(np.argmax(model.predict_proba(e)) == e.gold) for e in val_ex if e.gold >= 0])
            if acc > best:
                best, bad = acc, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
