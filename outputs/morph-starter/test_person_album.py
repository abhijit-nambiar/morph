import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote
import numpy as np
from PIL import Image
import morph

class Links(HTMLParser):
 def __init__(self):super().__init__();self.paths=[]
 def handle_starttag(self,tag,attrs):
  data=dict(attrs)
  if tag in ('a','img'):self.paths.append(data.get('href') or data.get('src'))

class PersonAlbumTest(unittest.TestCase):
 def test_two_scene_fit_preview_full_and_preflight(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);groups=[];masks=[]
   for scene in (1,2):
    photos=[]
    for i in (1,2):
     p=root/f's{scene}-{i}.png';rng=np.random.default_rng(scene*10+i);Image.fromarray(rng.integers(40,200,(40,50,3),dtype=np.uint8)).save(p);mask=root/f's{scene}-{i}-mask.png';m=np.zeros((40,50),dtype=np.uint8);m[8:35,10:35]=255;Image.fromarray(m).save(mask);photos.append(str(p.resolve()));masks.append({'source':str(p.resolve()),'source_sha256':morph.sha256_file(p),'mask':str(mask.resolve()),'mask_sha256':morph.sha256_file(mask)})
    groups.append({'id':f'scene-0{scene}','representative':photos[0],'photos':photos})
   groupsfile=root/'groups.json';groupsfile.write_text(json.dumps({'groups':groups}));maskfile=root/'masks.json';maskfile.write_text(json.dumps(masks));script=str(Path(__file__).with_name('person_album.py'));fits=root/'fits'
   subprocess.run([sys.executable,script,'fit',str(groupsfile),str(maskfile),'--prompt','warm and brighten people','--size','64','--out',str(fits)],check=True,capture_output=True)
   config=root/'corrections.json';config.write_text(json.dumps({'album':{'temperature':.03},'photos':{'scene-02':{'s2-2.png':{'exposure':.2}}}}));preview=root/'preview'
   subprocess.run([sys.executable,script,'apply',str(fits),'--preview-samples','1','--corrections',str(config),'--out',str(preview)],check=True,capture_output=True)
   report=json.loads((preview/'report.json').read_text());self.assertEqual(report['photos'],2);self.assertEqual(report['mode'],'preview');self.assertTrue((preview/'scene-02'/'exports'/'s2-1.png').exists())
   full=root/'full';subprocess.run([sys.executable,script,'apply',str(fits),'--corrections',str(config),'--out',str(full)],check=True,capture_output=True)
   scene=json.loads((full/'scene-02'/'report.json').read_text());self.assertEqual(scene['corrections']['photos']['s2-2.png']['exposure'],.2);self.assertTrue(Path(scene['photos'][1]['output']).exists());self.assertEqual(json.loads((full/'report.json').read_text())['photos'],4)
   review_script=str(Path(__file__).with_name('review_person_album.py'))
   for album,expected in ((preview,2),(full,4)):
    review=root/(album.name+'-review');subprocess.run([sys.executable,review_script,str(album),'--out',str(review),'--per-page','1'],check=True,capture_output=True)
    data=json.loads((review/'index.json').read_text());self.assertEqual(data['photos'],expected);self.assertEqual(data['missing_icc_profiles'],0)
    for page in review.rglob('*.html'):
     links=Links();links.feed(page.read_text())
     for link in links.paths:self.assertTrue((page.parent/unquote(link)).is_file(),link)
   album_report=json.loads((full/'report.json').read_text());album_report['photos']=99;(full/'report.json').write_text(json.dumps(album_report));invalid=root/'invalid-review'
   r=subprocess.run([sys.executable,review_script,str(full),'--out',str(invalid)],capture_output=True);self.assertNotEqual(r.returncode,0);self.assertFalse(invalid.exists());album_report['photos']=4;(full/'report.json').write_text(json.dumps(album_report))
   # A late unreadable export must not leave a partially successful review.
   Path(scene['photos'][1]['output']).write_bytes(b'broken image');broken=root/'broken-review'
   r=subprocess.run([sys.executable,review_script,str(full),'--out',str(broken)],capture_output=True);self.assertNotEqual(r.returncode,0);self.assertFalse(broken.exists())
   Path(masks[3]['mask']).write_bytes(b'changed');bad=root/'bad';r=subprocess.run([sys.executable,script,'apply',str(fits),'--preview-samples','1','--out',str(bad)],capture_output=True);self.assertNotEqual(r.returncode,0);self.assertFalse(bad.exists())
if __name__=='__main__':unittest.main()
