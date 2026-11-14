"""``morph`` command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _progress(label):
    t0 = time.time()

    def cb(i, n, name):
        print(f"\r{label} {i}/{n} {name[:40]:40s} {time.time() - t0:5.0f}s", end="" if i < n else "\n", file=sys.stderr, flush=True)

    return cb


def cmd_init(a):
    from .project import Project

    p = Project.create(a.project, a.album, a.cache, progress=_progress("analyse"), threshold=a.threshold)
    print(f"{len(p.album.photos)} photos, {len(p.scenes)} scenes -> {p.root}/project.json")
    for s in p.scenes:
        print(f"  {s.id}: {len(s.photos):3d} photos  representative {s.representative}")


def cmd_scenes(a):
    from .project import Project

    p = Project(a.project)
    scenes = p.recluster(a.threshold) if a.threshold else p.scenes
    print(json.dumps([s.to_json() for s in scenes], indent=1) if a.json else "\n".join(f"{s.id}: {len(s.photos)} photos, rep {s.representative}: {' '.join(s.photos)}" for s in scenes))


def cmd_suggest(a):
    from .project import Project

    for i, s in enumerate(Project(a.project).suggest(a.k), 1):
        print(f"{i}. {s.name:24s} {s.scene}  score {s.score:.2f}  - {'; '.join(s.reasons)}")


def cmd_reference(a):
    from .project import Project

    p = Project(a.project)
    if a.remove:
        p.remove_reference(a.photo)
        print("removed")
        return
    ref = p.reference_from_edit(a.photo, a.edit)
    print(f"reference {ref.name}: guard {ref.guard['verdict']}, fit dE {ref.fit['de_before']} -> {ref.fit['de_after']}")
    print("  " + "\n  ".join(ref.final.describe(20)))


def cmd_generate(a):
    from .project import Project

    p = Project(a.project)
    for i, (c, rep) in enumerate(p.generate(a.photo, a.prompt, a.generator, a.n)):
        print(f"[{i}] {c.label:14s} guard={rep.verdict:10s} edges={rep.edge_f1:.2f} structure={rep.structure_ok:.2f} {'; '.join(rep.reasons)}")
    print(f"candidates in {p.root}/candidates/ - choose with: morph pick {a.project} {a.photo} INDEX")


def cmd_pick(a):
    from .project import Project

    ref = Project(a.project).pick(a.photo, a.index)
    print(f"reference {ref.name} from {ref.source}")
    print("  " + "\n  ".join(ref.final.describe(20)))


def cmd_adjust(a):
    from .project import Project

    changes = {}
    for kv in a.changes:
        k, v = kv.split("=", 1)
        changes[k] = float(v)
    ref = Project(a.project).adjust(a.photo, changes)
    print("  " + "\n  ".join(ref.final.describe(30)))


def cmd_apply(a):
    from .export import export_album, review_page
    from .project import Project
    from .propagate import PropagationConfig, make_reference, propagate

    p = Project(a.project)
    refs = p.references()
    if not refs:
        sys.exit("no references yet: add one with `morph reference`, `morph generate` + `morph pick`, or `morph adjust`")
    cfg = PropagationConfig(harmonize_within=a.harmonize_within, harmonize_across=a.harmonize_across)
    R = [make_reference(p.album, r.name, r.final, p.scene_of(r.name)) for r in refs]
    recipes = propagate(p.album, R, p.scenes, cfg, progress=_progress("propagate"))
    report = export_album(p.album, recipes, a.out, a.max_side, a.quality, progress=_progress("export"))
    page = review_page(p.album, p.scenes, recipes, [r.name for r in refs], a.out)
    bad = [k for k, v in report.items() if not v["ok"]]
    print(f"exported {len(report)} photos to {a.out}/photos (structure check failures: {len(bad)}); review: {page}")


def cmd_edit(a):
    """One shot: analyse, pick references automatically, generate the look, export."""
    from .project import Project

    p = Project.create(a.out / "project", a.album, a.cache, progress=_progress("analyse"))
    sugg = p.suggest(a.references or None)
    for s in sugg[: a.references or len(sugg)]:
        if a.edit_dir:
            ed = next((f for f in Path(a.edit_dir).glob(Path(s.name).stem + ".*")), None)
            if ed is None:
                continue
            p.reference_from_edit(s.name, ed)
        else:
            cands = p.generate(s.name, a.look, a.generator, 1)
            if cands[0][1].verdict == "reject":
                print(f"skipping {s.name}: candidate rejected ({'; '.join(cands[0][1].reasons)})")
                continue
            p.pick(s.name, 0)
        print(f"reference: {s.name}")
    ns = argparse.Namespace(project=p.root, out=a.out, max_side=a.max_side, quality=a.quality, harmonize_within=0.5, harmonize_across=0.35)
    cmd_apply(ns)


def cmd_studio(a):
    from .studio import serve

    serve(a.project, a.host, a.port)


def cmd_evaluate(a):
    from .evaluate import evaluate

    s = evaluate(a.originals, a.edits, a.out, a.k, a.cache, a.refs, progress=_progress("analyse"))
    print(json.dumps(s, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="morph", description="Reference-driven, region-aware, non-structural album editing.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="analyse an album and group it into scenes")
    s.add_argument("project", type=Path); s.add_argument("album", type=Path)
    s.add_argument("--cache", type=Path); s.add_argument("--threshold", type=float, default=0.4)
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("scenes", help="show / re-cluster scenes")
    s.add_argument("project", type=Path); s.add_argument("--threshold", type=float); s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_scenes)

    s = sub.add_parser("suggest", help="rank photos that make good references")
    s.add_argument("project", type=Path); s.add_argument("-k", type=int)
    s.set_defaults(fn=cmd_suggest)

    s = sub.add_parser("reference", help="use your own edit of a photo as a reference")
    s.add_argument("project", type=Path); s.add_argument("photo"); s.add_argument("--edit", type=Path); s.add_argument("--remove", action="store_true")
    s.set_defaults(fn=cmd_reference)

    s = sub.add_parser("generate", help="generate candidate edits of a photo from a prompt")
    s.add_argument("project", type=Path); s.add_argument("photo"); s.add_argument("--prompt", required=True)
    s.add_argument("--generator", default="local", help="local | openai | https://... | files:DIR"); s.add_argument("-n", type=int, default=4)
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("pick", help="make a generated candidate the reference for its photo")
    s.add_argument("project", type=Path); s.add_argument("photo"); s.add_argument("index", type=int)
    s.set_defaults(fn=cmd_pick)

    s = sub.add_parser("adjust", help="manual tweaks on a reference: exposure=0.2 face_skin.texture=-0.3 ...")
    s.add_argument("project", type=Path); s.add_argument("photo"); s.add_argument("changes", nargs="+")
    s.set_defaults(fn=cmd_adjust)

    s = sub.add_parser("apply", help="propagate the references to the whole album and export")
    s.add_argument("project", type=Path); s.add_argument("--out", type=Path, required=True)
    s.add_argument("--max-side", type=int); s.add_argument("--quality", type=int, default=95)
    s.add_argument("--harmonize-within", type=float, default=0.5); s.add_argument("--harmonize-across", type=float, default=0.35)
    s.set_defaults(fn=cmd_apply)

    s = sub.add_parser("edit", help="one shot: album + look (or a folder of your edits) -> edited album")
    s.add_argument("album", type=Path); s.add_argument("--out", type=Path, required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--look"); g.add_argument("--edit-dir", type=Path, help="folder holding your edits of some photos (same file stems)")
    s.add_argument("--references", type=int, default=0, help="how many reference photos (default: one per scene, max 6)")
    s.add_argument("--generator", default="local"); s.add_argument("--cache", type=Path)
    s.add_argument("--max-side", type=int); s.add_argument("--quality", type=int, default=95)
    s.set_defaults(fn=cmd_edit)

    s = sub.add_parser("studio", help="local web UI for references, prompts and region tweaks")
    s.add_argument("project", type=Path); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8642)
    s.set_defaults(fn=cmd_studio)

    s = sub.add_parser("evaluate", help="benchmark against a hand-edited album")
    s.add_argument("originals", type=Path); s.add_argument("edits", type=Path); s.add_argument("--out", type=Path, required=True)
    s.add_argument("-k", type=int, default=3); s.add_argument("--refs", nargs="*"); s.add_argument("--cache", type=Path)
    s.set_defaults(fn=cmd_evaluate)

    a = ap.parse_args(argv)
    if os.environ.get("MORPH_THREADS"):  # avoid CPU oversubscription when several morph jobs share a machine
        import torch

        torch.set_num_threads(int(os.environ["MORPH_THREADS"]))
    a.fn(a)


if __name__ == "__main__":
    main()
