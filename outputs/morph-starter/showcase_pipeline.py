"""Portable evidence for the guided showcase; no fitting or correction runs here."""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import morph


def pipeline_data(root, groups, styles, normalization, evaluation):
    root = Path(root).resolve()
    scene_reports = {s['scene']: s for s in normalization['scenes']}
    measured = {p['source']: p for p in evaluation['photos']}
    verification = json.loads((root / 'verification.json').read_text())
    dimensions = {p['source']: p['dimensions'] for p in verification['photos']}
    result = {'scenes': {}, 'photos': {}, 'images': [], 'files': []}
    for group in groups:
        scene_id = group['id']
        report = scene_reports[scene_id]
        varied_ids = {measured[p]['varied']: Path(p).stem for p in group['photos']}
        edges = []
        for edge in report['edges']:
            edges.append({'left': varied_ids.get(edge['left'], 'reference'),
                          'right': varied_ids.get(edge['right'], 'reference'),
                          'matches': edge['geometric_inliers']})
        fits = {}
        for style in styles:
            directory = root / 'models' / style / 'fits' / scene_id
            fit = json.loads((directory / 'fit.json').read_text())
            recipe = fit['fitted_recipe']
            fits[style] = {'features': len(recipe['features']),
                           'coefficients': sum(len(row) for row in recipe['weights']),
                           'sampled_pixels': fit['metrics']['sampled_pixels'],
                           'rmse': fit['metrics']['rmse'],
                           'target': str((directory / 'target.jpg').relative_to(root)),
                           'recovered': str((directory / 'reconstruction.jpg').relative_to(root)),
                           'full_target': str((root / 'models' / style / 'targets' /
                                               Path(group['representative']).name).relative_to(root)),
                           'file': str((directory / 'fit.json').relative_to(root))}
            result['images'].extend([fits[style]['target'], fits[style]['recovered']])
            result['files'].extend([fits[style]['file'], fits[style]['full_target']])
        result['scenes'][scene_id] = {'edges': edges, 'fits': fits,
                                      'reference': Path(group['representative']).stem}
        for source in group['photos']:
            measurement = measured[source]
            varied = measurement['varied']
            correction = normalization['corrections'][varied]
            result['photos'][Path(source).stem] = {
                'correction': {k: correction[k] for k in ('scale', 'offset', 'status')},
                'matched_pairs': sum(varied in (edge['left'], edge['right']) for edge in report['edges']),
                'input': str(Path(varied).resolve().relative_to(root)),
                'dimensions': dimensions[varied],
                'normalization_before': measurement['normalization_before'],
                'normalization_after': measurement['normalization_after'],
                'styles': {key: {k: value[k] for k in ('before', 'after')}
                           for key, value in measurement['styles'].items()}}
            result['files'].append(result['photos'][Path(source).stem]['input'])
    result['files'].extend('final-albums/' + style + '/review.json' for style in styles)
    result['files'].extend(['groups.json', 'normalization-final.json', 'corrections-final.json',
                            'evaluation.json', 'verification.json'])
    return result


def attach_person_example(root, experiment, name=None, size=1200):
    """Explicitly copy one verified historical example, separate from album results."""
    root, experiment = Path(root).resolve(), Path(experiment).resolve()
    manifest = json.loads((experiment / 'masks.json').read_text())
    if name is None:
        groups = morph.validate_groups(json.loads((experiment / 'groups.json').read_text()))
        name = Path(max(groups, key=lambda g: len(g['photos']))['representative']).name
    item = next(row for row in manifest if Path(row['source']).name == name)
    for path_key, hash_key in (('source', 'source_sha256'), ('mask', 'mask_sha256')):
        if morph.sha256_file(item[path_key]) != item[hash_key]:
            raise ValueError('Historical person example changed: ' + item[path_key])
    # Scene reports contain the verified export path and its intended source.
    candidates = []
    for path in (experiment / 'full').glob('*/report.json'):
        candidates.extend(json.loads(path.read_text())['photos'])
    row = next(p for p in candidates if p['source'] == item['source'])
    source = morph.load(item['source'], size)
    with Image.open(item['mask']) as image:
        mask = np.asarray(image.convert('L').resize((source.shape[1], source.shape[0]),
                          Image.Resampling.BILINEAR), dtype=np.float32) / 255
    overlay = np.clip(source * (1 - .5 * mask[..., None]) +
                      np.array([1., .12, .08]) * .5 * mask[..., None], 0, 1)
    arrays = {'original': source, 'mask': np.repeat(mask[..., None], 3, axis=-1),
              'overlay': overlay, 'result': morph.load(row['output'], size)}
    images = {}
    for key, array in arrays.items():
        destination = root / 'assets' / ('local-person-' + key + '.jpg')
        if destination.exists():
            raise ValueError('Local example already exists: ' + str(destination))
        morph.save(array, destination, quality=93)
        images[key] = str(destination.relative_to(root))
    directory = root / 'local-experiment'
    directory.mkdir()
    destination = directory / name
    shutil.copy2(row['output'], destination)
    if morph.sha256_file(destination) != morph.sha256_file(row['output']):
        raise ValueError('Historical export copy differs')
    with Image.open(item['source']) as original, Image.open(destination) as exported:
        if exported.size != morph.oriented_size(original):
            raise ValueError('Historical export geometry differs')
        exported.load()
    return {'scope': 'Separate historical person-mask experiment; not part of the three styled albums. '
                     'Global warm/soft/shadow adjustments plus local person lighting. This mask covers '
                     'people including clothing, not isolated skin. The 13-photo run uses shared masks '
                     'for synthetic targets and recovery; low fitting error cannot validate masks.',
            'name': name, 'images': images, 'full': str(destination.relative_to(root)),
            'experiment_photos': len(candidates),
            'source_sha256': item['source_sha256'], 'mask_sha256': item['mask_sha256'],
            'export_sha256': morph.sha256_file(destination)}
