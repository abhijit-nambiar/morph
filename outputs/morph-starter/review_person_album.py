#!/usr/bin/env python3
"""Combine person-scene reviews into an album index with full-resolution links."""
import argparse
import html
import json
import os
import re
import tempfile
from pathlib import Path

import morph
import review_person_scene


def build_review(album, out, per_page=5):
    album = Path(album).resolve()
    out = Path(out).resolve()
    if out.exists() or not out.parent.is_dir():
        raise ValueError('Output already exists or parent is missing')
    if not 1 <= per_page <= 10:
        raise ValueError('per-page must be 1–10')
    report = json.loads((album / 'report.json').read_text())
    if report.get('kind') != 'experimental-person-album-output-v1':
        raise ValueError('Expected a person-album output report')
    scenes = report.get('scenes')
    if not isinstance(scenes, list) or not scenes:
        raise ValueError('Album contains no scenes')
    seen_scenes, seen_sources = set(), set()
    scene_reports = []
    for entry in scenes:
        scene = entry.get('scene')
        if not isinstance(scene, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', scene):
            raise ValueError('Invalid scene ID')
        if scene in seen_scenes or entry.get('report') != scene + '/report.json':
            raise ValueError('Duplicate scene or unexpected report path')
        seen_scenes.add(scene)
        path = album / scene / 'report.json'
        data = json.loads(path.read_text())
        rows = data.get('photos')
        if data.get('scene') != scene or not isinstance(rows, list) or not rows:
            raise ValueError('Invalid scene report')
        if entry.get('photos') != len(rows):
            raise ValueError('Scene photo count differs from album report')
        expected = set()
        for row in rows:
            source = Path(row['source']).resolve()
            output = Path(row['output']).resolve()
            if source in seen_sources:
                raise ValueError('Photo appears in multiple scene reports')
            seen_sources.add(source)
            if output != album / scene / 'exports' / source.name:
                raise ValueError('Export is outside its expected scene path')
            expected.add(output)
        actual = {p.resolve() for p in (album / scene / 'exports').iterdir() if p.is_file()}
        if actual != expected:
            raise ValueError('Missing or unexpected scene exports')
        scene_reports.append((scene, path))
    if report.get('photos') != len(seen_sources):
        raise ValueError('Album photo count differs from scene reports')
    if report.get('mode') not in ('preview', 'album'):
        raise ValueError('Invalid album mode')

    # Staging and final directories have the same depth, preserving relative
    # links from scene pages to exports when the completed review is renamed.
    with tempfile.TemporaryDirectory(prefix='.person-review-', dir=out.parent) as td:
        staged = Path(td)
        entries = []
        sections = []
        for scene, path in scene_reports:
            review_person_scene.build_review(path, staged / scene, per_page)
            data = json.loads((staged / scene / 'index.json').read_text())
            rows = data['photos']
            pages = sorted((staged / scene).glob('comparison-*.jpg'))
            entries.append({'scene': scene, 'review': scene + '/index.html', 'photos': rows})
            title = html.escape(scene)
            images = ''.join(
                f'<a href="{scene}/index.html"><img src="{scene}/{page.name}" '
                f'alt="{title}: original and edited photos"></a>' for page in pages
            )
            sections.append(
                f'<section><h2>{title} · {len(rows)} photos</h2>'
                f'<p><a href="{scene}/index.html">Open scene review and full-resolution exports</a></p>'
                f'{images}</section>'
            )
        photos = [row for scene in entries for row in scene['photos']]
        missing_icc = sum(not row['icc_profile_present'] for row in photos)
        flags = sum(row['review_flag'] for row in photos)
        scope = ('Export coverage, decodability and oriented dimensions checked; ICC presence reported. '
                 'Thumbnail comparisons do not validate masks, artistic quality or automatic normalization.')
        morph.dump({'scope': scope, 'mode': report['mode'], 'photos': len(photos),
                    'missing_icc_profiles': missing_icc, 'nearly_empty_masks': flags,
                    'scenes': entries}, staged / 'index.json')
        markup = (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Morph album review</title>'
            '<style>body{font:16px system-ui;max-width:1050px;margin:40px auto;'
            'padding:0 20px;background:#f5f5f3;color:#222}section{background:white;'
            'padding:16px;margin:24px 0;border-radius:8px}img{max-width:100%;height:auto}</style>'
            '<h1>Morph · scene edit transfer</h1>'
            f'<p>{len(entries)} scenes · {len(photos)} photos · {html.escape(report["mode"])} mode</p>'
            '<p>Each scene shares one recovered person-aware edit. Compare originals on the left '
            'with edited photos on the right; open a scene to inspect its full-resolution exports.</p>'
            f'<p>{html.escape(scope)}</p><p>Missing ICC profiles: {missing_icc}. '
            f'Nearly empty masks: {flags}. Inspect all mask overlays separately.</p>'
            + ''.join(sections) + '</html>'
        )
        (staged / 'index.html').write_text(markup, encoding='utf-8')
        os.rename(staged, out)
    print(len(photos), 'photos in', len(entries), 'scene reviews')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('album')
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-page', type=int, default=5)
    args = parser.parse_args()
    build_review(args.album, args.out, args.per_page)


if __name__ == '__main__':
    main()
