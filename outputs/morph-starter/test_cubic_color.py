import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import cubic_color
import morph
import poly_color


class CubicColorTest(unittest.TestCase):
    def test_independent_tone_curve_transfer_and_cli_replay(self):
        rng=np.random.default_rng(47)
        x=rng.uniform(.03,.97,(80,100,3)).astype(np.float32)
        holdout=rng.uniform(.03,.97,(70,90,3)).astype(np.float32)
        settings={'tone_contrast':-.4,'tone_shadows':.36,'tone_highlights':-.32}
        target=morph.pillow_soft_target(x,'neutral',settings)
        expected=morph.pillow_soft_target(holdout,'neutral',settings)
        quadratic,_=poly_color.fit(x,target,regularization=1e-5)
        cubic,_=cubic_color.fit(x,target,regularization=1e-6)
        self.assertLess(np.mean((cubic_color.render(holdout,cubic)-expected)**2),
                        np.mean((poly_color.render(holdout,quadratic)-expected)**2)*.5)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);photos=[]
            for i,image in enumerate((x,holdout)):
                path=root/f'{i}.png';Image.fromarray(np.uint8(image*255+.5)).save(path);photos.append(str(path))
            groups=root/'groups.json';groups.write_text(json.dumps({'groups':[
                {'id':'scene-01','photos':photos,'representative':photos[0]}]}))
            recipe=root/'target.json';recipe.write_text(json.dumps(settings))
            fit=root/'fit';script=str(Path(morph.__file__))
            subprocess.run([sys.executable,script,'album',str(groups),'--renderer','cubic',
                            '--solver','least-squares','--regularization','.000001',
                            '--target-engine','pillow-soft','--target-settings',str(recipe),
                            '--out',str(fit),'--size','100'],check=True,capture_output=True)
            replay=root/'replay'
            subprocess.run([sys.executable,script,'apply',str(groups),str(fit),
                            '--out',str(replay)],check=True,capture_output=True)
            for path in photos:
                left=np.asarray(Image.open(fit/'scene-01'/'exports'/Path(path).name))
                right=np.asarray(Image.open(replay/'scene-01'/'exports'/Path(path).name))
                np.testing.assert_array_equal(left,right)
            saved=json.loads((fit/'scene-01'/'fit.json').read_text())
            self.assertEqual(saved['fitted_recipe']['kind'],'cubic-rgb-v1')


if __name__=='__main__':unittest.main()
