import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import face_scene_pipeline as pipeline
import face_core as full
import morph,gamut_renderer,face_quality,audit_scene_consistency

class FaceSceneTests(unittest.TestCase):
    def test_prompt_validation(self):
        settings,local,_=pipeline.parse_prompt('warm and slightly brighten faces')
        self.assertEqual(settings['tone_temperature'],.12);self.assertEqual(local,.15)
        for prompt in ('warm','brighten faces and darken faces','brighten the person'):
            with self.assertRaises(ValueError):pipeline.parse_prompt(prompt)
        for box in ([],[{'x':0,'y':0,'width':2,'height':.2}],[{'x':float('nan'),'y':0,'width':.2,'height':.2}]):
            with self.assertRaises(ValueError):pipeline.checked_faces(box)

    def test_scene_brightness_proxy(self):
        x=np.full((40,50,3),.4,dtype=np.float32);box={'x':.2,'y':.2,'width':.2,'height':.2}
        value,n=audit_scene_consistency.face_brightness(x,[box]);self.assertAlmostEqual(value,.4,places=5);self.assertEqual(n,1)
        self.assertEqual(audit_scene_consistency.face_brightness(x,[]),(None,0))
        tiny={'x':.2,'y':.2,'width':.001,'height':.001};self.assertEqual(audit_scene_consistency.face_brightness(x,[tiny]),(None,0))

    def test_clipping_diagnostic_reports_unclamped_values(self):
        x=np.full((15,20,3),.4,dtype=np.float32);w=np.zeros((13,3));baseline=face_quality.diagnose(x,w,[])
        self.assertEqual(baseline['face_mapping_out_of_gamut_fraction'],0);self.assertFalse(baseline['near_limit_flag'])
        w[9,:]=1;extreme=face_quality.diagnose(x,w,[])
        self.assertEqual(extreme['face_mapping_out_of_gamut_fraction'],1);self.assertTrue(extreme['near_limit_flag'])
        low=face_quality.diagnose(x,w,[],photo={'exposure':-.2},scene={'contrast':.05},album={'temperature':.02})
        self.assertEqual(low['sample_size'],[20,15]);self.assertGreater(low['face_mapping_out_of_gamut_fraction'],0)

    def test_detector_wrapper_attaches_checksums(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);image=root/'photo.png';Image.new('RGB',(10,10),'white').save(image)
            groups=root/'groups.json';groups.write_text(json.dumps({'groups':[{'id':'scene-01','representative':str(image),'photos':[str(image)]}]}))
            detector=root/'fake_detector';detector.write_text('#!'+sys.executable+'\nimport json,sys\nprint(json.dumps([{"path":p,"faces":[]} for p in sys.argv[1:]]))\n');detector.chmod(0o755)
            wrapper=Path(pipeline.__file__).with_name('detect_scene_faces.py');out=root/'det.json'
            subprocess.run([sys.executable,str(wrapper),str(groups),'--detector',str(detector),'--out',str(out)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(out.read_text())[0]['source_sha256'],morph.sha256_file(image))
            review=Path(pipeline.__file__).with_name('review_face_detections.py');review_out=root/'review'
            subprocess.run([sys.executable,str(review),str(out),'--out',str(review_out)],check=True,capture_output=True,text=True);self.assertTrue((review_out/'detections-01.jpg').exists())
            additional=root/'additional.json';box={'x':.2,'y':.2,'width':.2,'height':.2};additional.write_text(json.dumps([{'path':str(image),'faces':[box],'source_sha256':morph.sha256_file(image)}]))
            merge=Path(pipeline.__file__).with_name('merge_face_detections.py');merged=root/'merged.json'
            subprocess.run([sys.executable,str(merge),str(out),str(additional),'--out',str(merged)],check=True,capture_output=True,text=True);self.assertEqual(json.loads(merged.read_text())[0]['faces'],[box])
            Image.new('RGB',(10,10),'red').save(image);bad=root/'stale-review'
            result=subprocess.run([sys.executable,str(review),str(out),'--out',str(bad)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())

    def test_no_detected_faces_uses_global_and_requires_face_representative(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);paths=[]
            for i in range(2):
                p=root/f'image{i}.png';Image.new('RGB',(40,50),(90,110,130)).save(p);paths.append(str(p))
            groups=root/'groups.json';groups.write_text(json.dumps({'groups':[{'id':'scene-01','representative':paths[0],'photos':paths}]}))
            box={'x':.2,'y':.3,'width':.15,'height':.2};det=root/'detections.json';rows=[{'path':p,'faces':[box] if i==0 else [],'source_sha256':morph.sha256_file(p)} for i,p in enumerate(paths)];det.write_text(json.dumps(rows));script=str(Path(pipeline.__file__));fit=root/'fit'
            subprocess.run([sys.executable,script,'fit',str(groups),str(det),'--scene','scene-01','--prompt','warm and brighten faces','--size','64','--out',str(fit)],check=True,capture_output=True,text=True)
            out=root/'out';subprocess.run([sys.executable,script,'apply',str(fit/'fit.json'),'--out',str(out)],check=True,capture_output=True,text=True)
            report=json.loads((out/'report.json').read_text());self.assertEqual(report['photos_without_detected_faces'],1);self.assertIn('skipped',report['photos'][1]['local_lighting'])
            weights=np.array(json.loads((fit/'fit.json').read_text())['weights']);x=morph.load(paths[1]);expected=full.model_tile(x,weights,[],0,x.shape)
            with Image.open(out/'exports'/'image1.png') as im:np.testing.assert_allclose(np.asarray(im,dtype=np.float32)/255,expected,atol=1/255)
            rows[0]['faces']=[];rows[1]['faces']=[box];det.write_text(json.dumps(rows));bad=root/'bad-representative'
            result=subprocess.run([sys.executable,script,'fit',str(groups),str(det),'--scene','scene-01','--prompt','warm and brighten faces','--out',str(bad)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())
            rows[1]['faces']=[];det.write_text(json.dumps(rows));empty=root/'no-faces-fit'
            subprocess.run([sys.executable,script,'fit',str(groups),str(det),'--scene','scene-01','--prompt','warm and brighten faces','--out',str(empty)],check=True,capture_output=True,text=True)
            data=json.loads((empty/'fit.json').read_text());self.assertIn('no faces detected',data['local_status']);np.testing.assert_array_equal(np.array(data['weights'])[10:],0)
            specific=root/'specific';subprocess.run([sys.executable,script,'apply',str(empty/'fit.json'),'--preview-photo','image1.png','--out',str(specific)],check=True,capture_output=True,text=True)
            selected=json.loads((specific/'report.json').read_text());self.assertEqual(selected['mode'],'preview');self.assertEqual(len(selected['photos']),1);self.assertEqual(Path(selected['photos'][0]['source']).name,'image1.png')
            for i,args in enumerate((['--preview-photo','missing.png'],['--preview-photo','image1.png','--preview-photo','image1.png'],['--preview-photo','image1.png','--preview-samples','1'])):
                rejected=root/f'bad-specific-{i}';result=subprocess.run([sys.executable,script,'apply',str(empty/'fit.json'),'--out',str(rejected),*args],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(rejected.exists())
            noface_out=root/'no-faces-out';subprocess.run([sys.executable,script,'apply',str(empty/'fit.json'),'--out',str(noface_out)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((noface_out/'report.json').read_text())['photos_without_detected_faces'],2)

    def test_album_scene_fits_and_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);groups=[];rows=[];box={'x':.2,'y':.2,'width':.2,'height':.2}
            for scene in (1,2):
                paths=[]
                for i in (1,2):
                    photo=root/f's{scene}-{i}.png';Image.new('RGB',(40,50),(80+scene*20,100+i*10,130)).save(photo);paths.append(str(photo));rows.append({'path':str(photo),'faces':[box],'source_sha256':morph.sha256_file(photo)})
                groups.append({'id':f'scene-0{scene}','representative':paths[0],'photos':paths})
            groupfile=root/'groups.json';groupfile.write_text(json.dumps({'groups':groups}));det=root/'detections.json';det.write_text(json.dumps(rows));script=str(Path(pipeline.__file__));fits=root/'fits'
            subprocess.run([sys.executable,script,'fit-album',str(groupfile),str(det),'--prompt','warm and brighten faces','--out',str(fits),'--size','64'],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((fits/'index.json').read_text())['scenes'],['scene-01','scene-02'])
            output=root/'preview';subprocess.run([sys.executable,script,'apply-album',str(fits),'--out',str(output),'--preview-samples','1'],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((output/'report.json').read_text())['mode'],'preview');self.assertTrue((output/'scene-02'/'exports'/'s2-1.png').exists())
            complete=root/'complete';subprocess.run([sys.executable,script,'apply-album',str(fits),'--out',str(complete)],check=True,capture_output=True,text=True)
            evaluation=root/'transfer.json';evaluator=Path(script).with_name('evaluate_face_transfer.py')
            subprocess.run([sys.executable,str(evaluator),str(fits),str(complete),'--out',str(evaluation)],check=True,capture_output=True,text=True)
            summary=json.loads(evaluation.read_text())['summary'];self.assertEqual(summary['rendered'],4);self.assertEqual(summary['other_photos'],2)
            consistency=root/'consistency.json';consistency_script=Path(script).with_name('audit_scene_consistency.py')
            subprocess.run([sys.executable,str(consistency_script),str(fits),str(complete),'--out',str(consistency)],check=True,capture_output=True,text=True)
            consistency_data=json.loads(consistency.read_text());self.assertEqual(consistency_data['summary']['photos'],4);self.assertTrue(all(x['status']=='insufficient measured photos' for x in consistency_data['scenes']))
            flag=root/'flag.json';flag.write_text(json.dumps({'photos':[{'scene':'scene-02','source':groups[1]['photos'][1],'export':str(complete/'scene-02'/'exports'/'s2-2.png'),'review_flag':True}]}))
            consistency_review=Path(script).with_name('review_scene_consistency.py');review_dir=root/'consistency-review'
            subprocess.run([sys.executable,str(consistency_review),'build',str(fits),str(complete),str(flag),'--out',str(review_dir)],check=True,capture_output=True,text=True)
            self.assertTrue((review_dir/'comparison-01.jpg').exists())
            decisions=review_dir/'decisions.csv';blank=root/'blank.json'
            subprocess.run([sys.executable,str(consistency_review),'compile',str(decisions),'--out',str(blank)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(blank.read_text()),{'photos':{}})
            import csv
            with decisions.open(newline='') as f: reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
            rows[0]['exposure']='0.2'
            with decisions.open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            selected=root/'selected.json';subprocess.run([sys.executable,str(consistency_review),'compile',str(decisions),'--out',str(selected)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(selected.read_text()),{'photos':{'scene-02':{'s2-2.png':{'exposure':.2}}}})
            selected_out=root/'selected-out';subprocess.run([sys.executable,script,'apply-album',str(fits),'--out',str(selected_out),'--preview-samples','2','--corrections',str(selected)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((selected_out/'scene-02'/'report.json').read_text())['corrections']['photos']['s2-2.png']['exposure'],.2)
            rows[0]['exposure']='nan'
            with decisions.open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            invalid_decision=root/'invalid-decision.json';result=subprocess.run([sys.executable,str(consistency_review),'compile',str(decisions),'--out',str(invalid_decision)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(invalid_decision.exists())
            invalid_review=root/'invalid-review';result=subprocess.run([sys.executable,str(consistency_review),'build',str(fits),str(selected_out),str(flag),'--out',str(invalid_review)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(invalid_review.exists())
            verify_script=Path(script).with_name('verify_face_album.py');verification=root/'verification.json'
            subprocess.run([sys.executable,str(verify_script),str(fits),str(complete),'--out',str(verification)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(verification.read_text())['expected'],4);self.assertEqual(json.loads(verification.read_text())['problem_count'],0)
            audit_script=Path(script).with_name('audit_face_album.py');audit_out=root/'clipping-audit.json'
            subprocess.run([sys.executable,str(audit_script),str(fits),str(complete),'--out',str(audit_out)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(audit_out.read_text())['photos'],4)
            overview=root/'album-review';review_album=Path(script).with_name('review_face_album.py')
            subprocess.run([sys.executable,str(review_album),str(complete),'--out',str(overview),'--per-page','2'],check=True,capture_output=True,text=True)
            review_index=json.loads((overview/'index.json').read_text());self.assertEqual(len(review_index['photos']),4);self.assertEqual(len(review_index['comparison_sheets']),2)
            (complete/'scene-02'/'exports'/'s2-2.png').unlink();failed=root/'failed-verification.json'
            result=subprocess.run([sys.executable,str(verify_script),str(fits),str(complete),'--out',str(failed)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertEqual(json.loads(failed.read_text())['problem_count'],1)
            corrections={'album':{'temperature':.04},'scenes':{'scene-02':{'contrast':.1}},'photos':{'scene-02':{'s2-2.png':{'exposure':.2}}}}
            config=root/'corrections.json';config.write_text(json.dumps(corrections));corrected=root/'corrected'
            subprocess.run([sys.executable,script,'apply-album',str(fits),'--out',str(corrected),'--preview-samples','2','--corrections',str(config)],check=True,capture_output=True,text=True)
            rec=json.loads((corrected/'scene-02'/'report.json').read_text());self.assertIn('face_mapping_out_of_gamut_fraction',rec['photos'][0]['clipping']);self.assertEqual(rec['operation_order'],['photo correction','fitted look','scene correction','album correction'])
            fit=json.loads((fits/'scene-02'/'fit.json').read_text());w=np.array(fit['weights']);x=morph.load(groups[1]['photos'][1]);x=gamut_renderer.render(x,gamut_renderer.vec({'exposure':.2}));x=full.model_tile(x,w,[box],0,x.shape);x=gamut_renderer.render(x,gamut_renderer.vec({'contrast':.1}));x=gamut_renderer.render(x,gamut_renderer.vec({'temperature':.04}))
            with Image.open(corrected/'scene-02'/'exports'/'s2-2.png') as image:np.testing.assert_allclose(np.asarray(image,dtype=np.float32)/255,x,atol=1/255)
            config.write_text(json.dumps({'photos':{'scene-02':{'missing.png':{'exposure':.2}}}}));badcorrection=root/'bad-correction'
            invalid=subprocess.run([sys.executable,script,'apply-album',str(fits),'--out',str(badcorrection),'--corrections',str(config)],capture_output=True,text=True);self.assertNotEqual(invalid.returncode,0);self.assertFalse(badcorrection.exists())
            suggest=Path(script).with_name('suggest_face_exposure.py');draft=root/'draft.json'
            subprocess.run([sys.executable,str(suggest),str(fits),'--out',str(draft)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(draft.read_text()),{'photos':{}})
            audit=root/'audit.json';audit.write_text(json.dumps({'photos':[{'scene':'scene-02','photo':groups[1]['photos'][1],'eligible':True,'proposed_exposure':.1}]}))
            review=root/'exposure-review';review_script=Path(script).with_name('review_face_exposure.py')
            subprocess.run([sys.executable,str(review_script),str(fits),str(audit),'--out',str(review)],check=True,capture_output=True,text=True);self.assertTrue((review/'comparison-01.jpg').exists())
            decisions=review/'decisions.csv';decisions.write_text(decisions.read_text().replace('0.1,','0.1,accept'))
            approved=root/'approved.json';subprocess.run([sys.executable,str(Path(script).with_name('review_exposure.py')),'compile',str(decisions),'--out',str(approved)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(approved.read_text()),{'photos':{'scene-02':{'s2-2.png':{'exposure':.1}}}})
            accepted=root/'accepted';subprocess.run([sys.executable,script,'apply-album',str(fits),'--out',str(accepted),'--preview-samples','2','--corrections',str(approved)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((accepted/'scene-02'/'report.json').read_text())['corrections']['photos']['s2-2.png']['exposure'],.1)
            Image.new('RGB',(40,50),'red').save(groups[1]['photos'][1]);bad=root/'bad-preview'
            result=subprocess.run([sys.executable,script,'apply-album',str(fits),'--out',str(bad),'--preview-samples','1'],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())
            badfit=root/'bad-fits';result=subprocess.run([sys.executable,script,'fit-album',str(groupfile),str(det),'--prompt','brighten faces','--out',str(badfit),'--size','64'],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(badfit.exists())

    def test_supplied_targets_fit_and_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);targets=root/'targets';targets.mkdir();groups=[];rows=[];box={'x':.2,'y':.2,'width':.2,'height':.2}
            for scene in (1,2):
                photo=root/f's{scene}.png';Image.new('RGB',(40,50),(90,110,130)).save(photo)
                target=targets/photo.name;Image.new('RGB',(40,50),(120,130,140)).save(target)
                groups.append({'id':f'scene-0{scene}','representative':str(photo),'photos':[str(photo)]});rows.append({'path':str(photo),'faces':[box],'source_sha256':morph.sha256_file(photo)})
            groupfile=root/'groups.json';groupfile.write_text(json.dumps({'groups':groups}));det=root/'detections.json';det.write_text(json.dumps(rows));script=str(Path(pipeline.__file__));fits=root/'fits'
            subprocess.run([sys.executable,script,'fit-album',str(groupfile),str(det),'--targets',str(targets),'--out',str(fits),'--size','64'],check=True,capture_output=True,text=True)
            fit=json.loads((fits/'scene-01'/'fit.json').read_text());self.assertEqual(fit['target_engine'],'supplied');self.assertEqual(fit['target_sha256'],morph.sha256_file(targets/'s1.png'));self.assertIsNone(fit['prompt'])
            out=root/'out';subprocess.run([sys.executable,script,'apply-album',str(fits),'--out',str(out),'--preview-samples','1'],check=True,capture_output=True,text=True);self.assertTrue((out/'scene-02'/'exports'/'s2.png').exists())
            Image.new('RGB',(20,50),'red').save(targets/'s2.png');bad=root/'bad-fits'
            result=subprocess.run([sys.executable,script,'fit-album',str(groupfile),str(det),'--targets',str(targets),'--out',str(bad),'--size','64'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())

    def test_fit_replay_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);paths=[];rng=np.random.default_rng(3)
            for i in range(2):
                p=root/f'image{i}.png';Image.fromarray(rng.integers(15,240,(70,80,3),dtype=np.uint8)).save(p);paths.append(str(p))
            groups=root/'groups.json';groups.write_text(json.dumps({'groups':[{'id':'scene-01','representative':paths[0],'photos':paths}]}))
            box={'x':.2,'y':.3,'width':.15,'height':.2};det=root/'detections.json';det.write_text(json.dumps([{'path':p,'faces':[box],'source_sha256':morph.sha256_file(p)} for p in paths]));fit=root/'fit'
            script=str(Path(pipeline.__file__))
            subprocess.run([sys.executable,script,'fit',str(groups),str(det),'--scene','scene-01','--prompt','warm and brighten faces','--size','64','--out',str(fit)],check=True,capture_output=True,text=True)
            data=json.loads((fit/'fit.json').read_text());self.assertEqual(len(data['manifest']),2);self.assertTrue(all(r['detection_validation']=='verified' for r in data['manifest']))
            stale=json.loads(det.read_text());stale[1]['source_sha256']='0'*64;det.write_text(json.dumps(stale));badfit=root/'badfit'
            failed=subprocess.run([sys.executable,script,'fit',str(groups),str(det),'--scene','scene-01','--prompt','brighten faces','--size','64','--out',str(badfit)],capture_output=True,text=True);self.assertNotEqual(failed.returncode,0);self.assertFalse(badfit.exists())
            out=root/'out';subprocess.run([sys.executable,script,'apply',str(fit/'fit.json'),'--out',str(out)],check=True,capture_output=True,text=True)
            w=np.array(data['weights']);expected=full.model_tile(morph.load(paths[1]),w,[box],0,morph.load(paths[1]).shape)
            with Image.open(out/'exports'/'image1.png') as image:np.testing.assert_allclose(np.asarray(image,dtype=np.float32)/255,expected,atol=1/255)
            Image.new('RGB',(80,70),'red').save(paths[1]);bad=root/'bad'
            result=subprocess.run([sys.executable,script,'apply',str(fit/'fit.json'),'--out',str(bad),'--preview-samples','1'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())

if __name__=='__main__':unittest.main()
