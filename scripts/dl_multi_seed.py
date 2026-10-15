# -*- coding: utf-8 -*-
"""
DL 多种子嵌套交叉验证（预热缓存）。

- 外层 StratifiedGroupKFold(5)：分组 = 归一化文本，防同文跨折泄漏
- 内层 GroupShuffleSplit(15%)：仅用于早停选轮，测试折绝不参与模型选择
- 5 个随机种子，输出 each-seed 的 OOF 概率
- 结果缓存到 Data/Comparison/dl_oof/
"""
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

PROJ = Path("/Users/Edward/Github/LLM-Sentiment-Analysis")
DATA = PROJ / "Data" / "Human" / "Human_Result.csv"
OUT = PROJ / "Data" / "Comparison" / "dl_oof"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 2024, 1234, 999]
N_FOLDS = 5
EPOCHS = 30
PATIENCE = 5
BATCH = 16
EMB = 128
HID = 128
MAXLEN = 128
DROPOUT = 0.3
LR = 1e-3
WD = 0.01
POS_WEIGHT = "sqrt"
INNER_VAL_FRAC = 0.15
DIMS = ("负面性", "侮辱性")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


def norm_key(t):
    s = unicodedata.normalize("NFKC", str(t))
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", s.lower())


class CharVocab:
    def __init__(self, maxlen=MAXLEN):
        self.maxlen = maxlen; self.vocab = {}
    def fit(self, texts):
        self.vocab = {c: i + 2 for i, c in enumerate(sorted({c for t in texts for c in t}))}
        return self
    @property
    def size(self): return len(self.vocab) + 2
    def encode(self, texts):
        ids = np.zeros((len(texts), self.maxlen), dtype=np.int64)
        msk = np.zeros((len(texts), self.maxlen), dtype=np.int64)
        for i, t in enumerate(texts):
            seq = [self.vocab.get(c, 1) for c in t[: self.maxlen]]
            ids[i, :len(seq)] = seq; msk[i, :len(seq)] = 1
            if not seq: msk[i, 0] = 1
        return torch.from_numpy(ids), torch.from_numpy(msk)


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.score = nn.Linear(d, 1)
    def forward(self, h, m):
        s = self.score(h).squeeze(-1).masked_fill(m == 0, -1e4)
        return torch.einsum("bl,blh->bh", torch.softmax(s, -1), h)


class Net(nn.Module):
    """字符级 BiGRU + 注意力池化 + 双 sigmoid 头。"""
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB, padding_idx=0)
        self.drop = nn.Dropout(DROPOUT)
        self.rnn = nn.GRU(EMB, HID // 2, batch_first=True, bidirectional=True)
        self.pool = AttnPool(HID)
        self.head = nn.Linear(HID, 2)
    def forward(self, ids, mask):
        x = self.drop(self.emb(ids))
        out, _ = self.rnn(x)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        return self.head(self.drop(self.pool(out, mask)))


class DS(Dataset):
    def __init__(self, texts, labels):
        self.t = list(texts); self.y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i], self.y[i]


def loader(enc, texts, labels, shuffle, gen=None):
    def collate(b):
        tx, ys = zip(*b)
        ids, msk = enc.encode(list(tx))
        return {"ids": ids, "mask": msk, "labels": torch.stack(ys)}
    return DataLoader(DS(texts, labels), batch_size=BATCH, shuffle=shuffle,
                      collate_fn=collate, num_workers=0, generator=gen)


@torch.no_grad()
def infer(model, ld):
    model.eval(); out = []
    for b in ld:
        out.append(torch.sigmoid(model(b["ids"].to(DEVICE), b["mask"].to(DEVICE))).cpu().numpy())
    return np.concatenate(out, 0)


def f1_np(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def train_one(seed, fold, tr_t, tr_y, va_t, va_y, te_t, te_y, groups_tr):
    """内层早停 + 返回测试折概率。Inner early stopping; returns test-fold probs."""
    set_seed(seed * 100 + fold)
    gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC, random_state=seed * 100 + fold)
    i_fit, i_val = next(gss.split(tr_t, tr_y[:, 1], groups_tr))
    fit_t = [tr_t[i] for i in i_fit]; fit_y = tr_y[i_fit]
    val_t = [tr_t[i] for i in i_val]; val_y = tr_y[i_val]

    enc = CharVocab().fit(fit_t)
    model = Net(enc.size).to(DEVICE)

    n_pos = fit_y.sum(0)
    w = np.sqrt((len(fit_y) - n_pos) / np.clip(n_pos, 1, None))
    crit = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w[k]], dtype=torch.float32).to(DEVICE))
            for k in range(2)]

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, math.ceil(len(fit_t) / BATCH)) * EPOCHS
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    tr_ld = loader(enc, fit_t, fit_y, True, torch.Generator().manual_seed(seed * 100 + fold))
    va_ld = loader(enc, val_t, val_y, False)

    best_score, best_state, best_ep, wait = -1.0, None, -1, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in tr_ld:
            y = b.pop("labels").to(DEVICE)
            ids, msk = b["ids"].to(DEVICE), b["mask"].to(DEVICE)
            logit = model(ids, msk)
            loss = crit[0](logit[:, 0], y[:, 0]) + crit[1](logit[:, 1], y[:, 1])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sch.last_epoch < steps - 1: sch.step()

        pv = infer(model, va_ld)
        sc = (f1_np(val_y[:, 0], (pv[:, 0] >= .5).astype(int))
              + f1_np(val_y[:, 1], (pv[:, 1] >= .5).astype(int))) / 2
        if sc > best_score + 1e-6:
            best_score, best_ep, wait = sc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    pt = infer(model, loader(enc, te_t, te_y, False))
    return pt, best_ep, best_score


def main():
    df = pd.read_csv(DATA)
    texts = df["评论"].astype(str).str.strip().tolist()
    Y = df[[f"{d}标签" for d in DIMS]].to_numpy(np.float32)
    groups = np.array([norm_key(t) for t in texts], dtype=object)
    strat = (Y[:, 0].astype(int) * 2 + Y[:, 1].astype(int))
    n = len(texts)

    print(f"device={DEVICE} | n={n} | seeds={SEEDS} | folds={N_FOLDS}")
    md5 = hashlib.md5(DATA.read_bytes()).hexdigest()
    print(f"data md5 = {md5}")

    all_rows = []
    for seed in SEEDS:
        t0 = time.time()
        oof = np.zeros((n, 2), dtype=float)
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (itr, ite) in enumerate(sgkf.split(np.zeros(n), strat, groups=groups), 1):
            te_t = [texts[i] for i in ite]; te_y = Y[ite]
            pt, bep, bsc = train_one(seed, fold,
                                     [texts[i] for i in itr], Y[itr],
                                     None, None, te_t, te_y, groups[itr])
            oof[ite] = pt
        dt = time.time() - t0
        f1n = f1_np(Y[:, 0], (oof[:, 0] >= .5).astype(int))
        f1i = f1_np(Y[:, 1], (oof[:, 1] >= .5).astype(int))
        print(f"  seed {seed:5d} | F1(负面)={f1n:.4f}  F1(侮辱)={f1i:.4f} | {dt:.1f}s")
        all_rows.append({"seed": seed, "F1_负面性": round(f1n, 4), "F1_侮辱性": round(f1i, 4),
                         "seconds": round(dt, 1)})
        np.save(OUT / f"oof_seed{seed}.npy", oof)

    pd.DataFrame(all_rows).to_csv(OUT / "seed_summary.csv", index=False, encoding="utf-8-sig")
    # 集成：5 seed 概率平均
    ens = np.mean([np.load(OUT / f"oof_seed{s}.npy") for s in SEEDS], axis=0)
    np.save(OUT / "oof_ensemble.npy", ens)
    meta = {"seeds": SEEDS, "data_md5": md5, "n_folds": N_FOLDS, "epochs": EPOCHS,
            "patience": PATIENCE, "batch": BATCH, "emb": EMB, "hidden": HID,
            "maxlen": MAXLEN, "dropout": DROPOUT, "lr": LR, "pos_weight": POS_WEIGHT,
            "inner_val_frac": INNER_VAL_FRAC, "arch": "char-bigru-attn",
            "torch": torch.__version__, "device": str(DEVICE)}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n集成 F1(负面)=%.4f  F1(侮辱)=%.4f" % (
        f1_np(Y[:, 0], (ens[:, 0] >= .5).astype(int)),
        f1_np(Y[:, 1], (ens[:, 1] >= .5).astype(int))))
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
