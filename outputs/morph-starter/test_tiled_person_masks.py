import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image
import morph

class TiledMasksTest(unittest.TestCase):
 def test_mask_replay_and_integrity(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/'source.png';oldmask=root/'old-mask.png';Image.new('RGB',(80,60),'blue').save(source);Image.new('L',(20,15),128).save(oldmask)
   entries=[{'source':str(source.resolve()),'source_sha256':morph.sha256_file(source),'mask':str(oldmask.resolve()),'mask_sha256':morph.sha256_file(oldmask)}];manifest=root/'manifest.json';manifest.write_text(json.dumps(entries))
   helper=root/'extract';helper.write_text('#!'+sys.executable+'\nimport sys\nfrom PIL import Image\nim=Image.open(sys.argv[1]);Image.new("L",im.size,128).save(sys.argv[2])\n');helper.chmod(0o755)
   script=str(Path(__file__).with_name('tiled_person_masks_experiment.py'));out=root/'output';subprocess.run([sys.executable,script,str(manifest),'--extractor',str(helper),'--out',str(out)],check=True,capture_output=True)
   result=json.loads((out/'manifest.json').read_text());self.assertEqual(result[0]['source_sha256'],entries[0]['source_sha256']);self.assertEqual(result[0]['mask_sha256'],morph.sha256_file(result[0]['mask']))
   with Image.open(result[0]['mask']) as im:self.assertEqual(im.size,(80,60));self.assertEqual(im.mode,'L')
   oldmask.write_bytes(b'changed');bad=root/'bad';r=subprocess.run([sys.executable,script,str(manifest),'--extractor',str(helper),'--out',str(bad)],capture_output=True);self.assertNotEqual(r.returncode,0);self.assertFalse(bad.exists())
if __name__=='__main__':unittest.main()
