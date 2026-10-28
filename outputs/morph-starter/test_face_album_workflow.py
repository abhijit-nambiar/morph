import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image
import morph,face_album_workflow

class WorkflowTests(unittest.TestCase):
    def test_original_folder_auto_groups_and_preview(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);images=root/'originals';images.mkdir();detections=[];box={'x':.2,'y':.2,'width':.2,'height':.2}
            for i in (1,2):
                p=images/f'{i}.png';Image.new('RGB',(50,40),(90,110,130)).save(p);detections.append({'path':str(p),'faces':[box],'source_sha256':morph.sha256_file(p)})
            det=root/'detections.json';det.write_text(json.dumps(detections));out=root/'run';script=str(Path(face_album_workflow.__file__))
            subprocess.run([sys.executable,script,'--originals',str(images),'--detections',str(det),'--prompt','brighten faces','--size','64','--preview-samples','1','--out',str(out)],check=True,capture_output=True,text=True)
            state=json.loads((out/'run.json').read_text());self.assertEqual(state['groups_origin'],'proposed chronological');self.assertEqual(len(json.loads((out/'groups.json').read_text())['groups']),2);self.assertTrue((out/'groups-review'/'index.json').exists());self.assertTrue((out/'review'/'index.html').exists())
            bad=root/'bad';result=subprocess.run([sys.executable,script,str(out/'groups.json'),'--originals',str(images),'--detections',str(det),'--prompt','brighten faces','--out',str(bad)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())

    def test_preview_full_and_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);groups=[];detections=[];box={'x':.2,'y':.2,'width':.2,'height':.2}
            for scene in (1,2):
                photos=[]
                for i in (1,2):
                    p=root/f'{scene}-{i}.png';Image.new('RGB',(50,40),(80+scene*20,110+i*10,150)).save(p);photos.append(str(p));detections.append({'path':str(p),'source_sha256':morph.sha256_file(p),'faces':[box]})
                groups.append({'id':f'scene-{scene}','representative':photos[0],'photos':photos})
            groupfile=root/'groups.json';groupfile.write_text(json.dumps({'groups':groups}));det=root/'detections.json';det.write_text(json.dumps(detections));script=str(Path(face_album_workflow.__file__))
            base=[sys.executable,script,str(groupfile),'--detections',str(det),'--prompt','warm and brighten faces','--size','64']
            preview=root/'preview';subprocess.run(base+['--out',str(preview),'--preview-samples','1'],check=True,capture_output=True,text=True)
            state=json.loads((preview/'run.json').read_text());self.assertEqual(state['status'],'complete');self.assertEqual(state['mode'],'preview');self.assertNotIn('verify',state['completed_stages']);self.assertTrue((preview/'review'/'index.html').exists());self.assertEqual(json.loads((preview/'transfer-evaluation.json').read_text())['summary']['rendered'],2);self.assertEqual(json.loads((preview/'scene-consistency.json').read_text())['summary']['photos'],2);self.assertEqual(json.loads((preview/'clipping-audit.json').read_text())['photos'],2)
            full=root/'full';subprocess.run(base+['--out',str(full),'--full'],check=True,capture_output=True,text=True)
            state=json.loads((full/'run.json').read_text());self.assertEqual(state['status'],'complete');self.assertEqual(json.loads((full/'verification.json').read_text())['expected'],4);self.assertEqual(json.loads((full/'transfer-evaluation.json').read_text())['summary']['other_photos'],2)
            bad=root/'bad';invalid=subprocess.run([sys.executable,script,str(groupfile),'--detections',str(det),'--prompt','remove person','--out',str(bad)],capture_output=True,text=True);self.assertNotEqual(invalid.returncode,0);self.assertFalse(bad.exists())
            bad_corr=root/'bad-correction';config=root/'corrections.json';config.write_text(json.dumps({'photos':{'scene-2':{'wrong.png':{'exposure':.1}}}}))
            invalid=subprocess.run(base+['--out',str(bad_corr),'--corrections',str(config)],capture_output=True,text=True);self.assertNotEqual(invalid.returncode,0);self.assertFalse(bad_corr.exists())
            Image.new('RGB',(50,40),'red').save(groups[1]['photos'][1]);stale=root/'stale';invalid=subprocess.run(base+['--out',str(stale)],capture_output=True,text=True)
            self.assertNotEqual(invalid.returncode,0);self.assertEqual(json.loads((stale/'run.json').read_text())['status'],'failed')

if __name__=='__main__':unittest.main()
