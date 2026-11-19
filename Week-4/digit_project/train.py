"""Train + evaluate. Run once:  python train.py

Phases
  0  data + ONE shared stratified split (70/15/15)
  1  BASELINE  : majority-class predictor
  2  V1 plain  : CNN and ViT, no augmentation
  3  V2 improve: CNN and ViT, controlled augmentation
  4  pick per model by VALIDATION score only; test set touched only to report
  5  save checkpoints (models/cnn.pt, models/vit.pt) + results/
"""
import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from common import (SEED, augment, build_model, load_data, make_split,
                    save_checkpoint)

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

CFG = {"cnn": dict(epochs=80, lr=3e-3, wd=1e-4, cfg={}),
       "vit": dict(epochs=150, lr=2e-3, wd=1e-2, cfg={})}
BATCH = 64


# ---------------------------------------------------------------- data
X, Y = load_data()
tr, va, te = make_split(Y)
assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))   # no leakage
json.dump({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()},
          open("results/split_indices.json", "w"))
Xtr, Ytr = torch.from_numpy(X[tr]), torch.from_numpy(Y[tr])
Xva, Yva = torch.from_numpy(X[va]), torch.from_numpy(Y[va])
Xte, Yte = torch.from_numpy(X[te]), torch.from_numpy(Y[te])
print(f"split train/val/test = {len(tr)}/{len(va)}/{len(te)}  pixel range {X.min():.2f}-{X.max():.2f}")

# fixed 'photo-like' perturbed copies of val/test (shift, rotate, blur, noise)
g = torch.Generator().manual_seed(123)
Xva_p, Xte_p = augment(Xva, g), augment(Xte, g)


def metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


@torch.no_grad()
def logits_of(model, x):
    model.eval()
    return model(x)


def pred_of(model, x):
    return logits_of(model, x).argmax(1).numpy()


# ---------------------------------------------------------------- phase 1 baseline
majority = int(np.bincount(Y[tr]).argmax())
base_pred = np.full(len(te), majority)
baseline = metrics(Y[te], base_pred)
print(f"BASELINE majority class = {majority}: {baseline}")


# ---------------------------------------------------------------- training fn
def train(name, use_aug):
    c = CFG[name]
    torch.manual_seed(SEED)
    model = build_model(name, c["cfg"])
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["wd"]) \
        if name == "cnn" else torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=c["lr"], total_steps=c["epochs"] * ((len(Xtr) + BATCH - 1) // BATCH))
    lossf = nn.CrossEntropyLoss()
    best, best_state, hist = (-1, 1e9), None, []
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(c["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            xb, yb = Xtr[b], Ytr[b]
            if use_aug:
                # half clean / half augmented so clean accuracy is not sacrificed
                half = len(xb) // 2
                xb = torch.cat([xb[:half], augment(xb[half:])])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        with torch.no_grad():
            model.eval()
            vl = lossf(model(Xva), Yva).item()
            vacc = (model(Xva).argmax(1) == Yva).float().mean().item()
            vpacc = (model(Xva_p).argmax(1) == Yva).float().mean().item()
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(Xtr), val_loss=vl,
                         val_acc=vacc, val_acc_perturbed=vpacc))
        key = ((vacc + vpacc) / 2, -vl)        # model selection uses VALIDATION only
        if best == (-1, 1e9) or key > best:
            best, best_state = key, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


# ---------------------------------------------------------------- phases 2-4
runs = {}
for name in ("cnn", "vit"):
    for variant, use_aug in (("plain", False), ("aug", True)):
        t = time.time()
        model, hist = train(name, use_aug)
        val_clean = metrics(Y[va], pred_of(model, Xva))
        val_pert = metrics(Y[va], pred_of(model, Xva_p))
        runs[(name, variant)] = dict(model=model, hist=hist, val_clean=val_clean,
                                     val_pert=val_pert,
                                     score=(val_clean["accuracy"] + val_pert["accuracy"]) / 2)
        print(f"{name:3s} {variant:5s} val acc clean {val_clean['accuracy']:.4f}  "
              f"perturbed {val_pert['accuracy']:.4f}  ({time.time()-t:.0f}s)")

final, report = {}, {"baseline": {"majority_class": majority, **baseline},
                     "split": {"train": len(tr), "val": len(va), "test": len(te)},
                     "variants": {}, "final": {}}
for name in ("cnn", "vit"):
    # pick by validation only; tie -> prefer augmented (targets real photos)
    pick = max(("plain", "aug"), key=lambda v: (round(runs[(name, v)]["score"], 4), v == "aug"))
    final[name] = pick
    for v in ("plain", "aug"):
        r = runs[(name, v)]
        yp, ypp = pred_of(r["model"], Xte), pred_of(r["model"], Xte_p)
        report["variants"][f"{name}_{v}"] = {
            "val_clean": r["val_clean"], "val_perturbed": r["val_pert"],
            "test_clean": metrics(Y[te], yp), "test_perturbed": metrics(Y[te], ypp)}
    r = runs[(name, pick)]
    yp = pred_of(r["model"], Xte)
    report["final"][name] = {"variant": pick, "test": metrics(Y[te], yp),
                             "test_perturbed": metrics(Y[te], pred_of(r["model"], Xte_p)),
                             "confusion_matrix": confusion_matrix(Y[te], yp, labels=range(10)).tolist()}
    save_checkpoint(f"models/{name}.pt", name, r["model"], CFG[name]["cfg"],
                    extra={"variant": pick, "split_seed": SEED,
                           "val_acc": r["val_clean"]["accuracy"]})
    json.dump(r["hist"], open(f"results/history_{name}_{pick}.json", "w"))
    print(f"FINAL {name}: {pick}  test {report['final'][name]['test']}")

json.dump(report, open("results/metrics.json", "w"), indent=2)

# ---------------------------------------------------------------- confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, name in zip(axs, ("cnn", "vit")):
    cm = np.array(report["final"][name]["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{name.upper()} ({final[name]}) acc "
                 f"{report['final'][name]['test']['accuracy']:.3f}")
plt.tight_layout(); plt.savefig("results/confusion_matrices.png", dpi=130); plt.close()

# ---------------------------------------------------------------- error examples + interpretation
preds = {n: pred_of(runs[(n, final[n])]["model"], Xte) for n in ("cnn", "vit")}
wrong = np.where((preds["cnn"] != Y[te]) | (preds["vit"] != Y[te]))[0][:10]
if len(wrong):
    fig, axs = plt.subplots(1, len(wrong), figsize=(1.6 * len(wrong), 2.2), squeeze=False)
    for ax, k in zip(axs[0], wrong):
        ax.imshow(X[te][k, 0], cmap="gray"); ax.axis("off")
        ax.set_title(f"T{Y[te][k]} C{preds['cnn'][k]} V{preds['vit'][k]}", fontsize=8)
    plt.suptitle("Test errors  (T true, C CNN, V ViT)", fontsize=9)
    plt.tight_layout(); plt.savefig("results/errors.png", dpi=130); plt.close()


def top_confusions(cm, k=3):
    off = [(cm[i][j], i, j) for i in range(10) for j in range(10) if i != j and cm[i][j] > 0]
    return sorted(off, reverse=True)[:k]


lines = ["# Results", "",
         f"Split (stratified, shared): train {len(tr)} / val {len(va)} / test {len(te)}", "",
         "## Test set (held out, used only for reporting)", "",
         "| Model | Accuracy | Macro F1 | Perturbed-test acc |", "|---|---|---|---|",
         f"| Majority baseline (always {majority}) | {baseline['accuracy']:.3f} | {baseline['macro_f1']:.3f} | - |"]
for n in ("cnn", "vit"):
    f = report["final"][n]
    lines.append(f"| {n.upper()} ({f['variant']}) | {f['test']['accuracy']:.3f} | "
                 f"{f['test']['macro_f1']:.3f} | {f['test_perturbed']['accuracy']:.3f} |")
lines += ["", "## Improvement path (plain -> augmented)", "",
          "| Variant | Val acc | Val perturbed | Test acc | Test perturbed |", "|---|---|---|---|---|"]
for k, v in report["variants"].items():
    lines.append(f"| {k} | {v['val_clean']['accuracy']:.3f} | {v['val_perturbed']['accuracy']:.3f} | "
                 f"{v['test_clean']['accuracy']:.3f} | {v['test_perturbed']['accuracy']:.3f} |")
lines += ["", "Perturbed = fixed random shift/rotation/scale/blur/noise copy of the same split,",
          "a proxy for messy real photos. Model variant chosen on VALIDATION only.", "",
          "## Mistakes", ""]
for n in ("cnn", "vit"):
    cm = np.array(report["final"][n]["confusion_matrix"])
    tc = top_confusions(cm)
    txt = ", ".join(f"true {i} -> pred {j} ({c}x)" for c, i, j in tc) or "none"
    lines.append(f"- {n.upper()} most common errors: {txt}.")
lines += ["- Only a handful of test errors, so no strong pattern; confusions involve digits whose 8x8 shapes "
          "look alike (e.g. 8 vs 1, 4 vs 6, 9 vs 0) because a stroke is only 1-2 pixels wide.",
          "- Real photos are harder than sklearn digits: different stroke width, lighting, "
          "background, blur and centering; resizing a big photo to 8x8 loses detail the models never saw. "
          "Matching the 0-1 scale does not guarantee photo accuracy.",
          "", "![cm](confusion_matrices.png)", "", "![err](errors.png)"]
open("results/results.md", "w").write("\n".join(lines))
print("\n".join(lines[:30]))
