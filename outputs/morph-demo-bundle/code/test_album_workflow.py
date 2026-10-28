import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image
import morph,album_workflow

class AlbumWorkflowTests(unittest.TestCase):
    def test_global_regional_and_face_routes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);paths=[];det=[];box={'x':.2,'y':.2,'width':.2,'height':.2}
            for i in range(2):
                p=root/f'image{i}.png';Image.new('RGB',(48,40),(90+i*15,120,140)).save(p);paths.append(str(p));det.append({'path':str(p),'source_sha256':morph.sha256_file(p),'faces':[box]})
            groups=root/'groups.json';groups.write_text(json.dumps({'groups':[{'id':'scene-01','representative':paths[0],'photos':paths}]}));detections=root/'det.json';detections.write_text(json.dumps(det));script=str(Path(album_workflow.__file__))
            base=[sys.executable,script,str(groups),'--size','64','--preview-samples','1']
            global_out=root/'global';subprocess.run(base+['--prompt','warm and lift shadows','--full','--out',str(global_out)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((global_out/'run.json').read_text())['renderer'],'gamut');self.assertEqual(json.loads((global_out/'verification.json').read_text())['photos'],2);self.assertTrue((global_out/'review'/'index.html').exists())
            verify=Path(script).with_name('verify_global_album.py');(global_out/'album'/'scene-01'/'exports'/'image1.png').unlink();badverify=root/'bad-verification.json'
            result=subprocess.run([sys.executable,str(verify),str(groups),str(global_out/'fits'),str(global_out/'album'),'--out',str(badverify)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertEqual(json.loads(badverify.read_text())['problem_count'],1)
            config=root/'regional-corrections.json';config.write_text(json.dumps({'local_strengths':{'scene-01':.5}}))
            region=root/'region';subprocess.run(base+['--prompt','warm and brighten upper left','--corrections',str(config),'--out',str(region)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((region/'run.json').read_text())['renderer'],'grid');self.assertEqual(json.loads((region/'album'/'report.json').read_text())['sample_count'],1)
            face=root/'face';subprocess.run(base+['--prompt','warm and brighten faces','--detections',str(detections),'--out',str(face)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((face/'run.json').read_text())['status'],'complete');self.assertTrue((face/'detections-review'/'index.json').exists())
            missing=root/'missing';result=subprocess.run(base+['--prompt','brighten faces','--out',str(missing)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(missing.exists())

if __name__=='__main__':unittest.main()
