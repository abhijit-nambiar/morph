import csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image
import edit_scene_groups

class SceneGroupTests(unittest.TestCase):
    def test_checked_split_and_reject_missing_duplicate_or_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);paths=[]
            for i in range(3):
                p=root/f'{i}.png';Image.new('RGB',(4,5)).save(p);paths.append(str(p))
            groups=root/'groups.json';groups.write_text(json.dumps({'groups':[{'id':'scene-01','representative':paths[0],'photos':paths}]}));draft=root/'assignments.csv';script=str(Path(edit_scene_groups.__file__))
            subprocess.run([sys.executable,script,'build',str(groups),'--out',str(draft)],check=True,capture_output=True,text=True)
            with open(draft,newline='') as f:rows=list(csv.DictReader(f))
            rows[2]['scene']='scene-02';rows[2]['representative']='yes'
            def write(items):
                with open(draft,'w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=edit_scene_groups.FIELDS);w.writeheader();w.writerows(items)
            write(rows);out=root/'new.json';subprocess.run([sys.executable,script,'compile',str(groups),str(draft),'--out',str(out)],check=True,capture_output=True,text=True)
            result=json.loads(out.read_text());self.assertEqual([len(g['photos']) for g in result['groups']],[2,1]);self.assertEqual(result['groups'][1]['representative'],str(Path(paths[2]).resolve()))
            for i,badrows in enumerate((rows[:-1],rows+[rows[0]],[dict(r,scene='../bad') if j==1 else r for j,r in enumerate(rows)],[dict(r,representative='yes') if j==1 else r for j,r in enumerate(rows)])):
                write(badrows);bad=root/f'bad{i}.json';r=subprocess.run([sys.executable,script,'compile',str(groups),str(draft),'--out',str(bad)],capture_output=True,text=True)
                self.assertNotEqual(r.returncode,0);self.assertFalse(bad.exists())

if __name__=='__main__':unittest.main()
