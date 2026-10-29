import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

import morph

try:
    import cv2
except ImportError:
    cv2=None


@unittest.skipIf(cv2 is None,'Optional OpenCV normalization dependency is unavailable')
class StyleWorkflowTest(unittest.TestCase):
    def test_repo_relative_fit_replays_after_relocation_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as td:
            source_root=Path(td).resolve()/'first-checkout';source_root.mkdir()
            originals=source_root/'originals';originals.mkdir()
            rng=np.random.default_rng(21)
            for i in range(2):
                Image.fromarray(rng.integers(30,210,(48,64,3),dtype=np.uint8)).save(originals/f'photo-{i}.png')
            groups={'groups':[{'id':'scene-01','representative':'originals/photo-0.png',
                               'photos':['originals/photo-0.png','originals/photo-1.png']}]}
            (source_root/'groups.json').write_text(json.dumps(groups))
            script=str(Path(morph.__file__).resolve())
            subprocess.run([sys.executable,script,'album','groups.json','--target-engine','pillow-soft',
                            '--renderer','cubic','--solver','least-squares','--out','fits'],
                           cwd=source_root,check=True,capture_output=True)
            fit_path=source_root/'fits/scene-01/fit.json';fit=json.loads(fit_path.read_text())
            for row in fit['source_manifest']:
                row['path']=str(Path(row['path']).relative_to(source_root))
            fit_path.write_text(json.dumps(fit))
            moved=Path(td)/'different-checkout';shutil.move(str(source_root),moved)
            command=[sys.executable,script,'apply','groups.json','fits','--out','replayed']
            subprocess.run(command,cwd=moved,check=True,capture_output=True)
            for i in range(2):
                with Image.open(moved/f'fits/scene-01/exports/photo-{i}.png') as before, \
                     Image.open(moved/f'replayed/scene-01/exports/photo-{i}.png') as after:
                    np.testing.assert_array_equal(np.asarray(before),np.asarray(after))
            Image.new('RGB',(64,48),(1,2,3)).save(moved/'originals/photo-1.png')
            changed=subprocess.run(command[:-1]+['changed'],cwd=moved,capture_output=True,text=True)
            self.assertNotEqual(changed.returncode,0)
            self.assertIn('file changed',changed.stderr)

    def test_complete_showcase_command_and_portable_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);originals=root/'originals';originals.mkdir()
            rng=np.random.default_rng(77)
            image=Image.new('RGB',(420,340),(95,115,130));draw=ImageDraw.Draw(image)
            for _ in range(140):
                x,y=rng.integers(0,380),rng.integers(0,300)
                draw.rectangle((int(x),int(y),int(x+30),int(y+30)),
                               fill=tuple(int(v) for v in rng.integers(35,220,3)))
            for i in range(5):image.save(originals/f'photo-{i}.png')
            scenes=root/'scenes.json';scenes.write_text(json.dumps({'groups':[
                {'id':'scene-01','label':'Test scene','representative':'photo-0.png',
                 'photos':[f'photo-{i}.png' for i in range(5)]}]}))
            styles=root/'styles.json';styles.write_text(json.dumps({'soft-film':{'name':'Soft film',
                'settings':{'tone_exposure':.12,'tone_contrast':-.4,'tone_shadows':.36,
                            'tone_highlights':-.32,'tone_temperature':.07,
                            'cool_saturation':-.35,'warm_saturation':-.22}}}))
            output=root/'showcase';script=Path(morph.__file__).with_name('showcase_workflow.py')
            process=subprocess.run([sys.executable,str(script),str(originals),'--scenes',str(scenes),
                                    '--styles',str(styles),'--size','320','--evaluation-size','512',
                                    '--out',str(output)],capture_output=True,text=True)
            if process.returncode:
                logs='\n'.join(p.read_text() for p in sorted((output/'logs').glob('*.log')))
                self.fail(process.stderr+'\n'+logs)
            self.assertEqual(json.loads((output/'run.json').read_text())['status'],'complete')
            verified=json.loads((output/'artifact-verification.json').read_text())
            self.assertEqual(verified['photos'],5);self.assertEqual(verified['problems'],0)
            self.assertEqual(json.loads((output/'verification.json').read_text())['exports'],6)
            data=json.loads((output/'showcase-data.json').read_text())
            self.assertEqual(len(data['photos']),5)
            self.assertTrue((output/'index.html').is_file())

    def test_variance_normalization_and_saved_style_replay_without_truth(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);image=Image.new('RGB',(420,340),(95,115,130));draw=ImageDraw.Draw(image)
            rng=np.random.default_rng(19)
            for _ in range(140):
                x,y=rng.integers(0,380),rng.integers(0,300)
                draw.rectangle((int(x),int(y),int(x+30),int(y+30)),
                               fill=tuple(int(v) for v in rng.integers(35,220,3)))
            paths=[]
            for i in range(2):
                path=root/f'photo-{i}.png';image.save(path);paths.append(str(path))
            groups=root/'groups.json';groups.write_text(json.dumps({'groups':[
                {'id':'scene-01','photos':paths,'representative':paths[0]}]}))
            settings={'tone_temperature':.2,'tone_exposure':.2,'tone_shadows':.2}
            styles=root/'styles.json';styles.write_text(json.dumps({'warm':{'name':'Warm','settings':settings}}))
            code=Path(morph.__file__).parent
            def run(script,*args):
                subprocess.run([sys.executable,str(code/script),*map(str,args)],check=True,capture_output=True)
            varied=root/'varied';run('make_variances.py',groups,'--seed','512','--out',varied)
            # The complete correction/replay path succeeds with evaluation truth
            # removed, demonstrating that the injected parameters are not inputs.
            (varied/'evaluation-truth.json').unlink()
            normal=root/'normal.json';corrections=root/'corrections.json'
            run('photometric_normalization.py',varied/'groups.json','--references',
                varied/'references.json','--size','512','--out',normal,'--corrections-out',corrections)
            models=root/'models';run('fit_style_album.py',groups,styles,'--size','128','--out',models)
            output=root/'album';run('apply_normalized_styles.py',varied/'groups.json',styles,models,
                                    normal,corrections,'--jobs','2','--out',output)
            self.assertEqual(json.loads((output/'run.json').read_text())['status'],'complete')
            self.assertEqual(len(json.loads((output/'warm/review.json').read_text())['photos']),3)
            target=morph.pillow_soft_target(morph.load(paths[1]),'neutral',settings)
            original=morph.load(varied/'inputs/varied-photo-1.png')
            actual=morph.load(output/'warm/scene-01/exports/varied-photo-1.png')
            self.assertLess(np.mean((actual-target)**2),np.mean((original-target)**2)*.15)
            replay_groups=json.loads((output/'replay-groups.json').read_text())
            self.assertEqual(replay_groups['groups'][0]['representative'],paths[0])
            self.assertEqual(actual.shape,target.shape)


if __name__=='__main__':unittest.main()
