import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image

class VisionClusterTest(unittest.TestCase):
 def test_average_linkage_and_invalid_matrix(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);photos=[]
   for i in range(4):
    f=root/f'{i}.png';Image.new('RGB',(8,8)).save(f);photos.append(str(f.resolve()))
   groups=root/'groups.json';groups.write_text(json.dumps({'groups':[{'id':'scene-01','representative':photos[0],'photos':photos}]}))
   distances=root/'distance.json';matrix=[[0,.1,.8,.8],[.1,0,.8,.8],[.8,.8,0,.2],[.8,.8,.2,0]];distances.write_text(json.dumps({'paths':photos,'distances':matrix,'revision':2}))
   script=str(Path(__file__).with_name('vision_cluster.py'));out=root/'out.json'
   subprocess.run([sys.executable,script,str(groups),str(distances),'--threshold','.5','--out',str(out)],check=True,capture_output=True)
   result=json.loads(out.read_text());self.assertEqual([len(x['photos']) for x in result['groups']],[2,2]);self.assertEqual({p for g in result['groups'] for p in g['photos']},set(photos))
   matrix[1][0]=.9;distances.write_text(json.dumps({'paths':photos,'distances':matrix}));bad=root/'bad.json'
   fail=subprocess.run([sys.executable,script,str(groups),str(distances),'--out',str(bad)],capture_output=True);self.assertNotEqual(fail.returncode,0);self.assertFalse(bad.exists())
if __name__=='__main__':unittest.main()
