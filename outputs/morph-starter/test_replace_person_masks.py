import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image
import morph

class ReplacementTest(unittest.TestCase):
 def test_explicit_replacement_and_integrity(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);src=root/'a.jpg';old=root/'old.png';new=root/'new.png';Image.new('RGB',(80,60),'blue').save(src);Image.new('L',(20,15),20).save(old);Image.new('L',(80,60),180).save(new)
   manifest=root/'manifest.json';manifest.write_text(json.dumps([{'source':str(src),'source_sha256':morph.sha256_file(src),'mask':str(old),'mask_sha256':morph.sha256_file(old)}]));script=str(Path(__file__).with_name('replace_person_masks.py'));out=root/'selected';base=[sys.executable,script,str(manifest),'--replace','a.jpg',str(new),'--out']
   subprocess.run(base+[str(out)],check=True,capture_output=True);row=json.loads((out/'manifest.json').read_text())[0];self.assertEqual(row['mask_sha256'],morph.sha256_file(new));self.assertEqual(morph.sha256_file(old),json.loads(manifest.read_text())[0]['mask_sha256'])
   with Image.open(row['mask']) as image:self.assertEqual(image.size,(80,60))
   invalid=root/'invalid';result=subprocess.run([sys.executable,script,str(manifest),'--replace','missing.jpg',str(new),'--out',str(invalid)],capture_output=True);self.assertNotEqual(result.returncode,0);self.assertFalse(invalid.exists())
   old.write_bytes(b'changed');invalid2=root/'invalid2';result=subprocess.run(base+[str(invalid2)],capture_output=True);self.assertNotEqual(result.returncode,0);self.assertFalse(invalid2.exists())
if __name__=='__main__':unittest.main()
