"""Reads the executed Phase 2 notebook and writes every reported number to reports/phase2_results.json,
plus the notebook's figures to reports/figures/nb_*.png. The report and slides are built from this file,
so no number is typed by hand.

Run from the repo root:  python reports/scripts/extract_phase2_results.py [notebook.ipynb]
"""
import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "02_phase2_deep_learning.ipynb"
nb = json.load(open(nb_path))

texts, images = [], []
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        if "text" in o:
            texts.append("".join(o["text"]))
        d = o.get("data", {})
        if "text/plain" in d and "Figure" not in "".join(d["text/plain"]):
            texts.append("".join(d["text/plain"]))
        if "image/png" in d:
            images.append(d["image/png"])
out = "\n".join(texts)

PM = r"([\d.]+) ± ([\d.]+)"
res = {"cv": {}, "test": {}, "err_by_bin": {}, "params": {}, "best_epochs": {}, "cv_seconds": {}}

# CV table (mean ± std per metric)
cv_block = out.split("scored on uncapped RUL (mean ± std across folds)")[1].split("\n\n")[0]
for line in cv_block.strip().splitlines()[1:]:
    m = re.match(r"(.+?)\s{2,}" + r"\s+".join([PM] * 4) + r"\s*$", line)
    if m:
        v = list(map(float, m.groups()[1:]))
        res["cv"][m.group(1).strip()] = {k: {"mean": v[2 * i], "std": v[2 * i + 1]}
                                         for i, k in enumerate(["RMSE", "MAE", "MAE early", "MAE late"])}

# Best epochs and timing per CV run
for m in re.finditer(r"^(.+?)\s+RMSE\s+[\d.]+ ± [\d.]+ \| MAE\s+[\d.]+ \| best epochs \[([\d, ]+)\] \|\s+(\d+)s", out, re.M):
    res["best_epochs"][m.group(1).strip()] = [int(x) for x in m.group(2).split(",")]
    res["cv_seconds"][m.group(1).strip()] = int(m.group(3))

# Parameter counts
for m in re.finditer(r"^(\w+)\s+parameters: ([\d,]+)", out, re.M):
    res["params"][m.group(1)] = int(m.group(2).replace(",", ""))

# Test table
tb = out.split("deep models: mean ± std over 3 seeds)")[1].split("\n\n")[0].strip().splitlines()
for line in tb[1:]:
    name = re.split(r"\s{2,}", line.strip())[0]
    rest = line.strip()[len(name):]
    pms = re.findall(PM, rest)
    keys = ["RMSE", "MAE", "MAE early", "MAE late"]
    if pms:
        tail = re.findall(r"[\d.]+", re.sub(PM, "", rest))
        res["test"][name] = {k: {"mean": float(a), "std": float(b)} for k, (a, b) in zip(keys, pms)}
        res["test"][name]["ens_RMSE"], res["test"][name]["ens_MAE"] = float(tail[0]), float(tail[1])
        res["test"][name]["epochs"] = int(float(tail[2]))
    else:
        nums = list(map(float, re.findall(r"[\d.]+", rest)))
        res["test"][name] = {k: {"mean": v, "std": 0.0} for k, v in zip(keys, nums)}

m = re.search(r"Early engines \(true RUL >= 50\): (\d+) \| late engines: (\d+)", out)
res["n_test_early"], res["n_test_late"] = int(m.group(1)), int(m.group(2))

# Error by RUL bin
eb = out.split("Hybrid LSTM + features  rows")[1].split("\n\n")[0].strip().splitlines()
for line in eb:
    p = line.split()
    res["err_by_bin"][p[0]] = {"rf": float(p[1]), "lstm": float(p[2]), "hybrid": float(p[3]), "rows": int(p[4])}

m = re.search(r"numpy ([\S]+) \| pandas ([\S]+) \| scikit-learn ([\S]+) \| torch ([\S]+)", out)
res["versions"] = dict(zip(["numpy", "pandas", "scikit-learn", "torch"], m.groups()))

# Figures, in notebook order
names = ["nb_learning_curves", "nb_ablation_rmse", "nb_error_analysis", "nb_test_predictions"]
figdir = ROOT / "reports" / "figures"
for n, img in zip(names, images):
    (figdir / f"{n}.png").write_bytes(base64.b64decode(img))

json.dump(res, open(ROOT / "reports" / "phase2_results.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))
print("images:", len(images))
