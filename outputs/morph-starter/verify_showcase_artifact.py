#!/usr/bin/env python3
"""Check portable showcase data, every preview image and every local file link."""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlparse

from PIL import Image

import morph


class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        data=dict(attrs)
        if tag=='a' and data.get('href'):self.paths.append(data['href'])
        if tag=='img' and data.get('src'):self.paths.append(data['src'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('--out',required=True);args=parser.parse_args()
    root=Path(args.root).resolve();data=json.loads((root/'showcase-data.json').read_text())
    photos=data['photos'];ids={p['id'] for p in photos}
    if len(ids)!=len(photos):raise ValueError('Duplicate photo IDs')
    assigned=[p for scene in data['scenes'] for p in scene['photos']]
    if len(set(assigned))!=len(assigned) or set(assigned)!=ids:raise ValueError('Scene coverage differs')
    expected=len(json.loads((root/'varied-album/evaluation-truth.json').read_text())['photos'])
    if len(photos)!=expected:raise ValueError('Showcase photo coverage differs from input album')
    images=set();files=set()
    for photo in photos:
        images.update(photo['assets'][key] for key in ('source','varied','normalized'))
        for style in data['styles']:
            images.update(photo['assets']['styles'][style['id']].values())
            files.add(photo['full'][style['id']])
    images.add(data['reference']['source'])
    for style in data['reference']['styles'].values():images.update(style.values())
    if 'pipeline' in data:
        images.update(data['pipeline']['images'])
        files.update(data['pipeline']['files'])
    if data.get('localEdits'):
        images.update(data['localEdits']['images'].values())
        files.add(data['localEdits']['full'])
    links=Links();markup=(root/'index.html').read_text();links.feed(markup)
    for path in links.paths:
        if path.startswith('#'):continue
        if urlparse(path).scheme:raise ValueError('Showcase has an external dependency: '+path)
        files.add(unquote(path))
    if '__MORPH_DATA__' in markup:raise ValueError('Template data was not bound')
    for relative in images|files:
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Missing or nonportable file: '+relative)
    for relative in images:
        with Image.open(root/relative) as image:image.load()
    if data.get('localEdits') and morph.sha256_file(root/data['localEdits']['full']) != data['localEdits']['export_sha256']:
        raise ValueError('Separate local experiment export changed')
    morph.protect_existing([args.out],False)
    morph.dump({'scope':'Scene coverage, all dynamic preview images, all full-resolution links, '
                'static HTML links and local portability checked. Browser interactions and visual '
                'layout require separate QA.','photos':len(photos),'preview_images':len(images),
                'full_resolution_links':len(photos)*len(data['styles']),
                'separate_local_exports':int(bool(data.get('localEdits'))),
                'files_checked':len(images|files),'problems':0},args.out)
    print(len(photos),'photos;',len(images),'preview images;',len(images|files),'local files checked',flush=True)


if __name__=='__main__':main()
