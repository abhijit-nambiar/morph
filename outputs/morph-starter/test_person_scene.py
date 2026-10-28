import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import morph,person_scene,gamut_renderer

class PersonSceneTest(unittest.TestCase):
 def test_fit_apply_and_changed_mask_preflight(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);photos=[];rows=[]
   for i in range(3):
    rng=np.random.default_rng(40+i);x=rng.integers(40,200,(70,90,3),dtype=np.uint8);p=root/f'{i}.png';Image.fromarray(x).save(p);mask=root/f'{i}-mask.png';m=np.zeros((70,90),dtype=np.uint8);m[15:55,20+i*5:55+i*5]=255;Image.fromarray(m).save(mask);photos.append(str(p.resolve()));rows.append({'source':str(p.resolve()),'source_sha256':morph.sha256_file(p),'mask':str(mask.resolve()),'mask_sha256':morph.sha256_file(mask)})
   group=root/'groups.json';group.write_text(json.dumps({'groups':[{'id':'scene-01','representative':photos[0],'photos':photos}]}));masks=root/'masks.json';masks.write_text(json.dumps(rows));script=str(Path(person_scene.__file__));fit=root/'fit'
   subprocess.run([sys.executable,script,'fit',str(group),str(masks),'--scene','scene-01','--prompt','warm and brighten people','--size','128','--out',str(fit)],check=True,capture_output=True)
   output=root/'preview';subprocess.run([sys.executable,script,'apply',str(fit/'fit.json'),'--preview-photo','1.png','--out',str(output)],check=True,capture_output=True)
   with Image.open(output/'exports'/'1.png') as img:self.assertEqual(img.size,(90,70))
   report=json.loads((output/'report.json').read_text());self.assertEqual(report['mode'],'preview');self.assertFalse(report['photos'][0]['review_flag'])
   scene_review=root/'scene-review';scene_review_script=str(Path(person_scene.__file__).with_name('review_person_scene.py'))
   subprocess.run([sys.executable,scene_review_script,str(output/'report.json'),'--out',str(scene_review)],check=True,capture_output=True)
   scene_data=json.loads((scene_review/'index.json').read_text());self.assertEqual(len(scene_data['photos']),1);self.assertTrue(scene_data['photos'][0]['icc_profile_present']);self.assertTrue((scene_review/'comparison-01.jpg').exists())
   review=root/'mask-review';review_script=str(Path(person_scene.__file__).with_name('review_person_masks.py'))
   subprocess.run([sys.executable,review_script,str(masks),'--out',str(review),'--per-page','2'],check=True,capture_output=True)
   review_data=json.loads((review/'index.json').read_text());self.assertEqual(len(review_data['photos']),3);self.assertTrue((review/'comparison-02.jpg').exists());self.assertTrue((review/'index.html').exists())
   x=morph.load(photos[1]);m=person_scene.mask_array(rows[1]['mask'],x.shape);settings,local,_=person_scene.parse_prompt('warm and brighten people');y=morph.pillow_soft_target(x,'neutral',settings);target=np.clip(y+m[...,None]*local*y*(1-y),0,1)
   reconstructed=morph.load(output/'exports'/'1.png');self.assertLess(np.mean((reconstructed-target)**2),np.mean((x-target)**2)*.35)
   corrections={'album':{'temperature':.04},'scenes':{'scene-01':{'contrast':.1}},'photos':{'scene-01':{'1.png':{'exposure':.2}}}}
   config=root/'corrections.json';config.write_text(json.dumps(corrections));corrected=root/'corrected'
   subprocess.run([sys.executable,script,'apply',str(fit/'fit.json'),'--preview-photo','1.png','--corrections',str(config),'--out',str(corrected)],check=True,capture_output=True)
   w=np.asarray(json.loads((fit/'fit.json').read_text())['weights']);adjusted=gamut_renderer.render(x,gamut_renderer.vec({'exposure':.2}));expected=np.clip(adjusted+person_scene.features(adjusted,m)@w,0,1);expected=gamut_renderer.render(expected,gamut_renderer.vec({'contrast':.1}));expected=gamut_renderer.render(expected,gamut_renderer.vec({'temperature':.04}))
   with Image.open(corrected/'exports'/'1.png') as image:np.testing.assert_allclose(np.asarray(image,dtype=np.float32)/255,expected,atol=1/255)
   data=json.loads((corrected/'report.json').read_text());self.assertEqual(data['operation_order'],['photo correction','fitted look','scene correction','album correction']);self.assertEqual(data['corrections']['photos']['1.png']['exposure'],.2)
   config.write_text(json.dumps({'photos':{'scene-01':{'missing.png':{'exposure':.2}}}}));unknown=root/'unknown';r=subprocess.run([sys.executable,script,'apply',str(fit/'fit.json'),'--corrections',str(config),'--out',str(unknown)],capture_output=True);self.assertNotEqual(r.returncode,0);self.assertFalse(unknown.exists())
   Path(rows[2]['mask']).write_bytes(b'changed');bad=root/'bad';r=subprocess.run([sys.executable,script,'apply',str(fit/'fit.json'),'--preview-photo','1.png','--out',str(bad)],capture_output=True);self.assertNotEqual(r.returncode,0);self.assertFalse(bad.exists())
   badreview=root/'bad-review';r=subprocess.run([sys.executable,review_script,str(masks),'--out',str(badreview)],capture_output=True);self.assertNotEqual(r.returncode,0);self.assertFalse(badreview.exists())
 def test_prompt_rejects_ambiguous(self):
  for value in ('warm','brighten people and darken people','do not brighten people'):
   with self.assertRaises(ValueError):person_scene.parse_prompt(value)
if __name__=='__main__':unittest.main()
