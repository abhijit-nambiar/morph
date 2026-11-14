"""Build a realistic labelled camera dump from a folder of good photos, and score `morph cull` on it.

    python scripts/make_cull_benchmark.py Morph/originals runs/cull-bench          # build + evaluate
    python scripts/make_cull_benchmark.py Morph/originals runs/cull-bench --eval-only

Every source photo becomes a scene (keeper). Most scenes also get failed or redundant frames
shot "a second later": defocus, focus on the background instead of the people, camera shake,
badly under/over-exposed frames, and near-duplicate burst frames (small shifts/rotation/exposure
jitter + sensor noise). Some scenes replace the original with a mildly imperfect version that a
photographer would still keep (slightly dark/bright, noisy, a touch soft), to measure false rejects.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from morph import cull
from morph.imageio import list_images, load_rgb, read_info

SIDE = 3000  # dump resolution (long side); blur sizes below are in pixels at cull.MEASURE_SIDE
K = SIDE / cull.MEASURE_SIDE


def _lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _enc(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def disc(img, r):
    r = max(1, round(r * K))
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1, -1)
    return cv2.filter2D(img, -1, k / k.sum())


def shake(img, length, angle):
    L = max(3, round(length * K)) | 1
    k = np.zeros((L, L), np.float32)
    c = L // 2
    dx, dy = np.cos(np.radians(angle)) * c, np.sin(np.radians(angle)) * c
    cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1, cv2.LINE_AA)
    return cv2.filter2D(img, -1, k / k.sum())


def exposure(img, ev):
    return _enc(_lin(img) * 2.0**ev).astype(np.float32)


def noise(img, rng, sigma):
    return np.clip(img + rng.normal(0, sigma, img.shape).astype(np.float32), 0, 1)


def jitter(img, rng):
    h, w = img.shape[:2]
    a = rng.uniform(-0.6, 0.6)
    s = 1 + rng.uniform(0.0, 0.02)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), a, s)
    m[:, 2] += rng.uniform(-0.012, 0.012, 2) * (w, h)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return noise(exposure(out, rng.uniform(-0.15, 0.15)), rng, 0.004)


def person_mask(img, seg):
    from morph.segment import HUMAN_MODEL, _group_probs, _HUMAN_GROUPS

    h, w = img.shape[:2]
    s = 512 / max(h, w)
    small = (cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    probs, labels = seg._probs(HUMAN_MODEL, small)
    p = 1 - _group_probs(probs, labels, _HUMAN_GROUPS)["background"]
    m = cv2.resize(p, (w, h))
    m = cv2.dilate((m > 0.5).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 12)[..., None]


def save(img, path, src, when):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    exif = Image.open(src).getexif()
    exif[0x0112] = 1
    stamp = when.strftime("%Y:%m:%d %H:%M:%S")
    exif[0x0132] = stamp
    exif.get_ifd(0x8769)[0x9003] = stamp
    pil.save(path, quality=92, exif=exif.tobytes())


def build(src_dir: Path, out: Path, seed: int = 0):
    from morph.segment import Segmenter

    seg = Segmenter()
    rng = np.random.default_rng(seed)
    pr = random.Random(seed)
    dump = out / "dump"
    dump.mkdir(parents=True, exist_ok=True)
    truth = {}
    files = list_images(src_dir)
    kinds = ["defocus", "subject_miss", "shake", "under", "over", "dup", "dup2", "mild", "none"]
    plan = [kinds[i % len(kinds)] for i in range(len(files))]
    pr.shuffle(plan)
    for i, (f, kind) in enumerate(zip(files, plan)):
        info = read_info(f, with_hash=False)
        t0 = info.captured
        img = load_rgb(f, max_side=SIDE)
        stem = f"SC{i:03d}"
        frames = []  # (suffix, image, label, kind)
        if kind == "mild":
            mild = pr.choice(["dark", "bright", "noisy", "soft"])
            base = {"dark": lambda: exposure(img, -0.9), "bright": lambda: exposure(img, 0.6),
                    "noisy": lambda: noise(img, rng, 0.02), "soft": lambda: disc(img, 1)}[mild]()
            frames.append(("a", base, "keep", f"mild_{mild}"))
        else:
            frames.append(("a", img, "keep", "original"))
        extra = [kind] if kind not in ("mild", "none") else []
        if kind in ("defocus", "shake", "under") and pr.random() < 0.5:
            extra.append("dup")  # failed frame inside a real burst
        for j, e in enumerate(extra):
            if e == "defocus":
                frames.append(("x", disc(img, pr.uniform(3.5, 7)), "reject", "defocus"))
            elif e == "subject_miss":
                m = person_mask(img, seg)
                if m.mean() < 0.03:
                    frames.append(("x", disc(img, pr.uniform(3.5, 6)), "reject", "defocus"))
                else:
                    frames.append(("x", img * (1 - m) + disc(img, pr.uniform(3.5, 6)) * m, "reject", "subject_miss"))
            elif e == "shake":
                frames.append(("x", shake(img, pr.uniform(10, 24), pr.uniform(0, 180)), "reject", "shake"))
            elif e == "under":
                frames.append(("x", noise(exposure(img, -pr.uniform(2.8, 3.8)), rng, 0.006), "reject", "under"))
            elif e == "over":
                frames.append(("x", exposure(img, pr.uniform(2.2, 3.2)), "reject", "over"))
            elif e in ("dup", "dup2"):
                for _ in range(2 if e == "dup2" else 1):
                    frames.append(("d", jitter(img, rng), "dup", "near_duplicate"))
        for k, (suf, im, label, what) in enumerate(frames):
            name = f"{stem}_{k}{suf}.jpg"
            save(im, dump / name, f, t0 + timedelta(seconds=k))
            truth[name] = {"label": label, "kind": what, "scene": i, "source": f.name}
        print(f"{i + 1}/{len(files)} {f.name}: {[w for *_, w in frames]}", flush=True)
    (out / "truth.json").write_text(json.dumps(truth, indent=1))


def evaluate(out: Path, settings: cull.CullSettings | None = None, people: bool = True, verbose: bool = True) -> dict:
    truth = json.loads((out / "truth.json").read_text())
    scores = cull.score_folder(out / "dump", cache=out / "cache", people=people,
                               progress=lambda i, n, name: print(f"\rscored {i}/{n}", end="", file=sys.stderr))
    print(file=sys.stderr)
    return metrics(truth, cull.decide(scores, settings or cull.CullSettings()), verbose)


def metrics(truth: dict, decisions, verbose: bool = True) -> dict:
    keep = {d.name: d.keep for d in decisions}
    why = {d.name: "; ".join(d.reasons) for d in decisions}
    tp = fp = fn = tn = 0  # positive = "should reject"
    per_kind: dict[str, list[int]] = {}
    errors = []
    scenes: dict[int, list[str]] = {}
    for n, t in truth.items():
        scenes.setdefault(t["scene"], []).append(n)
    for sc, names in scenes.items():
        sharp = [n for n in names if truth[n]["label"] in ("keep", "dup")]  # interchangeable good frames
        k = sum(keep[n] for n in sharp)
        good = len(sharp)
        tn += min(k, 1)
        fn += max(0, k - 1)
        tp += min(good - k, good - 1)
        fp += 1 if k == 0 else 0
        if k == 0:
            errors.append(f"lost scene {sc}: " + ", ".join(f"{n} ({why[n]})" for n in sharp))
        if k > 1:
            errors.append(f"kept {k} near-duplicates in scene {sc}: " + ", ".join(n for n in sharp if keep[n]))
        for n in names:
            t = truth[n]
            if t["label"] == "reject":
                ok = not keep[n]
                tp += ok
                fn += not ok
                if not ok:
                    errors.append(f"missed {t['kind']}: {n} ({why[n]})")
            ok = (not keep[n]) if t["label"] == "reject" else (k == 1 if t["label"] == "dup" else keep[n] or k >= 1)
            per_kind.setdefault(t["kind"], [0, 0])
            per_kind[t["kind"]][0] += int(ok)
            per_kind[t["kind"]][1] += 1
    res = {
        "frames": len(truth), "scenes": len(scenes),
        "reject_precision": round(tp / max(1, tp + fp), 3), "reject_recall": round(tp / max(1, tp + fn), 3),
        "keep_precision": round(tn / max(1, tn + fn), 3), "keep_recall": round(tn / max(1, tn + fp), 3),
        "per_kind_accuracy": {k: f"{a}/{b}" for k, (a, b) in sorted(per_kind.items())},
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    if verbose:
        print(json.dumps(res, indent=1))
        for e in errors:
            print(" -", e)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-people", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.eval_only:
        build(a.src, a.out, a.seed)
    evaluate(a.out, people=not a.no_people)
