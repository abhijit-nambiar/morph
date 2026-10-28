import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image

class VisionGroupTest(unittest.TestCase):
 def test_group_review_and_source_integrity(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);originals=root/'originals';originals.mkdir()
   for n in range(3):
    ex=Image.Exif();ex[306]='2026:01:01 12:00:00';Image.new('RGB',(32,24),(80+n*30,90,100)).save(originals/f'{n}.jpg',exif=ex)
   helper=root/'extractor';helper.write_text('#!'+sys.executable+'\nimport json,sys\np=json.load(open(sys.argv[1]));print(json.dumps({"paths":p,"distances":[[0 if i==j else .2 for j in range(len(p))] for i in range(len(p))],"revision":2}))\n');helper.chmod(0o755)
   script=str(Path(__file__).with_name('vision_group.py'));out=root/'result';run=[sys.executable,script,str(originals),'--extractor',str(helper),'--out',str(out)]
   subprocess.run(run,check=True,capture_output=True);result=json.loads((out/'groups.json').read_text());self.assertEqual(len(result['groups']),1);self.assertEqual(len(result['source_manifest_sha256']),3);self.assertTrue((out/'review'/'index.html').exists());self.assertEqual(result['source_groups'],'chronological.json')
   album=root/'album';workflow=str(Path(__file__).with_name('album_workflow.py'))
   subprocess.run([sys.executable,workflow,str(out/'groups.json'),'--prompt','warm and lift shadows','--size','64','--preview-samples','1','--out',str(album)],check=True,capture_output=True)
   self.assertEqual(json.loads((album/'run.json').read_text())['status'],'complete');self.assertTrue((album/'review'/'index.html').exists())
   self.assertNotEqual(subprocess.run(run,capture_output=True).returncode,0)
   helper.write_text('#!'+sys.executable+'\nimport json,sys\np=json.load(open(sys.argv[1]));open(p[0],"ab").write(b"x");print(json.dumps({"paths":p,"distances":[[0]*len(p) for _ in p]}))\n');helper.chmod(0o755);bad=root/'changed'
   result=subprocess.run([sys.executable,script,str(originals),'--extractor',str(helper),'--out',str(bad)],capture_output=True);self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())
if __name__=='__main__':unittest.main()
