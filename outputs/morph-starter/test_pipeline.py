import json, subprocess, sys, tempfile, unittest, csv
from pathlib import Path
import numpy as np
from PIL import Image
import morph
import selective
import gamut_renderer
import poly_color
import solver
import prompt_target
import suggest_exposure
import review_exposure
import spatial_light
import grid_light
from datetime import datetime,timedelta

class PipelineTests(unittest.TestCase):
    def test_reconstruct_and_transfer_from_independent_editor(self):
        rng=np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); images=[]
            for i in range(2):
                p=root/f'image{i}.png'; Image.fromarray(rng.integers(20,220,(80,100,3),dtype=np.uint8)).save(p); images.append(str(p))
            group={'groups':[{'id':'scene-01','representative':images[0],'photos':images}]}
            groupfile=root/'groups.json';groupfile.write_text(json.dumps(group));out=root/'result'
            cmd=[sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--target-engine','pillow','--preset','warm','--out',str(out),'--size','100']
            subprocess.run(cmd,check=True,capture_output=True,text=True)
            report=json.loads((out/'scene-01'/'fit.json').read_text())
            self.assertEqual(report['target_engine'],'pillow'); self.assertIsNone(report['known_recipe'])
            review=json.loads((out/'review.json').read_text())
            self.assertEqual(len(review['photos']),2)
            self.assertEqual(json.loads((out/'report.json').read_text())['export_count'],2)
            self.assertLess(report['metrics']['fit_mse'],report['metrics']['initial_mse']*.15)
            fitted=morph.vec(report['fitted_recipe']); holdout=morph.load(images[1],100)
            rmse=np.sqrt(np.mean((morph.render(holdout,fitted)-morph.pillow_target(holdout,'warm'))**2))
            self.assertLess(rmse,.04)
            with Image.open(out/'scene-01'/'exports'/'image1.png') as result:
                self.assertEqual(result.size,(100,80))
            targets=root/'targets';targets.mkdir()
            ref=morph.pillow_target(morph.load(images[0],100),'warm')
            Image.fromarray(np.uint8(ref*255+.5)).save(targets/'image0.png')
            output2=root/'file-target'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--targets',str(targets),'--out',str(output2),'--size','100'],check=True,capture_output=True,text=True)
            details=json.loads((output2/'scene-01'/'fit.json').read_text())
            self.assertEqual(details['target_engine'],'file')
            self.assertLess(details['metrics']['fit_mse'],details['metrics']['initial_mse']*.15)
            output3=root/'zero-strength'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--target-engine','pillow','--strength','0','--out',str(output3),'--size','100'],check=True,capture_output=True,text=True)
            zero=json.loads((output3/'scene-01'/'fit.json').read_text())
            self.assertTrue(any(v!=0 for v in zero['fitted_recipe'].values()))
            self.assertTrue(all(v==0 for v in zero['applied_recipe'].values()))
            np.testing.assert_array_equal(np.asarray(Image.open(images[1])),np.asarray(Image.open(output3/'scene-01'/'exports'/'image1.png')))
            recipe_file=root/'selective.json';recipe_file.write_text(json.dumps({'exposure':.1,'cool_saturation':.3,'warm_saturation':-.2}))
            output4=root/'selective'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','selective','--solver','gauss-newton','--recipe',str(recipe_file),'--out',str(output4),'--size','100'],check=True,capture_output=True,text=True)
            selected=json.loads((output4/'scene-01'/'fit.json').read_text())
            self.assertEqual(selected['renderer'],'selective');self.assertLess(selected['metrics']['rmse'],.001)
            self.assertLess(abs(selected['fitted_recipe']['cool_saturation']-.3),.02)
            self.assertLess(abs(selected['fitted_recipe']['warm_saturation']+.2),.02)
            with Image.open(output4/'scene-01'/'exports'/'image1.png') as result:self.assertEqual(result.size,(100,80))
            output5=root/'gamut'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','gamut','--solver','gauss-newton','--recipe',str(recipe_file),'--out',str(output5),'--size','100'],check=True,capture_output=True,text=True)
            bounded=json.loads((output5/'scene-01'/'fit.json').read_text())
            self.assertEqual(bounded['renderer'],'gamut');self.assertLess(bounded['metrics']['rmse'],.001)
            self.assertLess(abs(bounded['fitted_recipe']['cool_saturation']-.3),.02)
            with Image.open(output5/'scene-01'/'exports'/'image1.png') as result:self.assertEqual(result.size,(100,80))
            replay=root/'replay'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(output5),'--out',str(replay)],check=True,capture_output=True,text=True)
            with Image.open(output5/'scene-01'/'exports'/'image1.png') as original_render, Image.open(replay/'scene-01'/'exports'/'image1.png') as replay_render:
                np.testing.assert_array_equal(np.asarray(original_render),np.asarray(replay_render))
            corrections={'album':{'temperature':.04},'scenes':{'scene-01':{'contrast':.08}},'photos':{'scene-01':{'image1.png':{'exposure':.2}}}}
            correction_file=root/'corrections.json';correction_file.write_text(json.dumps(corrections));corrected=root/'corrected'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(output5),'--out',str(corrected),'--strength','0','--corrections',str(correction_file)],check=True,capture_output=True,text=True)
            original=morph.load(images[1]);expected=gamut_renderer.render(original,gamut_renderer.vec({'exposure':.2}));expected=gamut_renderer.render(expected,gamut_renderer.vec({'contrast':.08}));expected=gamut_renderer.render(expected,gamut_renderer.vec({'temperature':.04}))
            with Image.open(corrected/'scene-01'/'exports'/'image1.png') as result:np.testing.assert_allclose(np.asarray(result,dtype=np.float32)/255,expected,atol=1/255)
            saved=json.loads((corrected/'scene-01'/'applied.json').read_text());self.assertEqual(saved['operation_order'],['photo correction','fitted look','scene correction','album correction'])
            correction_preview=root/'correction-preview'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(output5),'--out',str(correction_preview),'--strength','0','--corrections',str(correction_file),'--preview-samples','2'],check=True,capture_output=True,text=True)
            preview=json.loads((correction_preview/'report.json').read_text());self.assertEqual(preview['mode'],'preview');self.assertEqual(preview['export_count'],0);self.assertEqual(preview['sample_count'],2)
            with Image.open(correction_preview/'scene-01'/'samples'/'image1.png') as sample,Image.open(corrected/'scene-01'/'exports'/'image1.png') as full:
                np.testing.assert_array_equal(np.asarray(sample),np.asarray(full))
            corrections['photos']['scene-01']['missing.jpg']={'exposure':.2};correction_file.write_text(json.dumps(corrections))
            bad_correction=subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(output5),'--out',str(root/'bad-correction'),'--corrections',str(correction_file)],capture_output=True,text=True)
            self.assertNotEqual(bad_correction.returncode,0)
            altered={'groups':[dict(group['groups'][0],representative=images[1])]};groupfile.write_text(json.dumps(altered))
            bad=subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(output5),'--out',str(root/'wrong-rep')],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);groupfile.write_text(json.dumps(group))
            output6=root/'independent-complex'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','selective','--solver','gauss-newton','--target-engine','pillow-selective','--out',str(output6),'--size','100'],check=True,capture_output=True,text=True)
            complex_fit=json.loads((output6/'scene-01'/'fit.json').read_text())
            self.assertEqual(complex_fit['target_engine'],'pillow-selective');self.assertEqual(complex_fit['renderer'],'selective')
            self.assertLess(complex_fit['metrics']['fit_mse'],complex_fit['metrics']['initial_mse'])
            source=morph.load(images[0],100)
            self.assertGreater(float(np.max(abs(morph.pillow_target(source,'warm')-morph.pillow_selective_target(source,'warm')))),.01)
            output7=root/'soft-target'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','gamut','--solver','gauss-newton','--target-engine','pillow-soft','--out',str(output7),'--size','100'],check=True,capture_output=True,text=True)
            soft=json.loads((output7/'scene-01'/'fit.json').read_text())
            self.assertEqual(soft['target_engine'],'pillow-soft');self.assertLess(soft['metrics']['fit_mse'],soft['metrics']['initial_mse'])
            settings=root/'target.json';settings.write_text(json.dumps({'tone_exposure':.15,'tone_shadows':.2,'warm_saturation':.2}))
            output8=root/'custom-soft'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','gamut','--solver','gauss-newton','--target-engine','pillow-soft','--target-settings',str(settings),'--full-res-reference','--out',str(output8),'--size','100'],check=True,capture_output=True,text=True)
            custom=json.loads((output8/'scene-01'/'fit.json').read_text())
            self.assertEqual(custom['target_settings']['tone_exposure'],.15);self.assertEqual(custom['target_settings']['tone_shadows'],.2)
            self.assertLess(custom['metrics']['fit_mse'],custom['metrics']['initial_mse'])
            for filename in ('target-full.jpg','reconstruction-full.jpg','applied-full.jpg'):
                with Image.open(output8/'scene-01'/filename) as im:self.assertEqual(im.size,(100,80))
            output9=root/'polynomial'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','polynomial','--solver','least-squares','--regularization','.001','--target-engine','pillow-soft','--out',str(output9),'--size','100'],check=True,capture_output=True,text=True)
            poly=json.loads((output9/'scene-01'/'fit.json').read_text())
            self.assertEqual(poly['fitted_recipe']['kind'],'polynomial-rgb-v1');self.assertLess(poly['metrics']['fit_mse'],poly['metrics']['initial_mse'])
            poly_review=json.loads((output9/'review.json').read_text());self.assertIn('polynomial_out_of_gamut_fraction',poly_review['photos'][0]['clipping'])
            poly_replay=root/'polynomial-replay'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(output9),'--out',str(poly_replay)],check=True,capture_output=True,text=True)
            with Image.open(output9/'scene-01'/'exports'/'image1.png') as first,Image.open(poly_replay/'scene-01'/'exports'/'image1.png') as replayed:np.testing.assert_array_equal(np.asarray(first),np.asarray(replayed))
            poly_corrections={'album':{'temperature':.04},'scenes':{'scene-01':{'contrast':.08}},'photos':{'scene-01':{'image1.png':{'exposure':.2}}}}
            poly_correction_file=root/'poly-corrections.json';poly_correction_file.write_text(json.dumps(poly_corrections));poly_corrected=root/'poly-corrected'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(output9),'--out',str(poly_corrected),'--corrections',str(poly_correction_file)],check=True,capture_output=True,text=True)
            expected=gamut_renderer.render(morph.load(images[1]),gamut_renderer.vec({'exposure':.2}));expected=poly_color.render(expected,poly_color.deserialize(poly['fitted_recipe']));expected=gamut_renderer.render(expected,gamut_renderer.vec({'contrast':.08}));expected=gamut_renderer.render(expected,gamut_renderer.vec({'temperature':.04}))
            with Image.open(poly_corrected/'scene-01'/'exports'/'image1.png') as actual:np.testing.assert_allclose(np.asarray(actual,dtype=np.float32)/255,expected,atol=1/255)
            applied=json.loads((poly_corrected/'scene-01'/'applied.json').read_text());self.assertEqual(applied['correction_renderer'],'gamut');self.assertEqual(applied['renderer'],'polynomial')
            prompt_out=root/'prompted'
            prompt='warm, slightly soft contrast and lift shadows'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','gamut','--solver','gauss-newton','--target-engine','pillow-soft','--prompt',prompt,'--out',str(prompt_out),'--size','100'],check=True,capture_output=True,text=True)
            prompt_fit=json.loads((prompt_out/'scene-01'/'fit.json').read_text())
            self.assertEqual(prompt_fit['prompt'],prompt);self.assertEqual(prompt_fit['target_settings']['tone_temperature'],.12);self.assertEqual(prompt_fit['target_settings']['tone_contrast'],-.1)
            preview_out=root/'prompt-preview'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','gamut','--solver','gauss-newton','--target-engine','pillow-soft','--prompt',prompt,'--preview-only','--full-res-reference','--out',str(preview_out),'--size','100'],check=True,capture_output=True,text=True)
            preview_report=json.loads((preview_out/'report.json').read_text());self.assertEqual(preview_report['mode'],'preview');self.assertEqual(preview_report['export_count'],0);self.assertIsNone(preview_report['clipping_flags'])
            self.assertFalse((preview_out/'scene-01'/'exports').exists());self.assertTrue((preview_out/'scene-01'/'target-full.jpg').exists())
            preview_apply=root/'preview-apply'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(preview_out),'--out',str(preview_apply)],check=True,capture_output=True,text=True)
            self.assertTrue((preview_apply/'scene-01'/'exports'/'image1.png').exists())
            sampled=root/'sample-preview'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','gamut','--solver','gauss-newton','--target-engine','pillow-soft','--prompt',prompt,'--preview-only','--preview-samples','2','--evaluate-transfer','--out',str(sampled),'--size','100'],check=True,capture_output=True,text=True)
            sample_report=json.loads((sampled/'report.json').read_text());self.assertEqual(sample_report['sample_count'],2);self.assertEqual(sample_report['export_count'],0)
            self.assertEqual(sample_report['transfer_check']['other_photos_evaluated'],1)
            transfer_rows=json.loads((sampled/'review.json').read_text())['photos'];self.assertEqual(len(transfer_rows),2)
            self.assertEqual({row['transfer']['role'] for row in transfer_rows},{'representative','other photo'});self.assertTrue(all(row['transfer']['status']=='evaluated' for row in transfer_rows))
            missing_targets=root/'missing-heldout-target'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','gamut','--solver','gauss-newton','--targets',str(targets),'--preview-only','--preview-samples','2','--evaluate-transfer','--out',str(missing_targets),'--size','100'],check=True,capture_output=True,text=True)
            missing_rows=json.loads((missing_targets/'review.json').read_text())['photos'];missing=next(x for x in missing_rows if x['transfer']['role']=='other photo')
            self.assertEqual(missing['transfer']['status'],'unavailable');self.assertEqual(missing['transfer']['reason'],'no same-name target')
            spatial_preview=root/'spatial-preview'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','spatial','--solver','least-squares','--regularization','.0001','--target-engine','spatial-soft','--target-local-light','.3','--preview-only','--preview-samples','2','--evaluate-transfer','--full-res-reference','--out',str(spatial_preview),'--size','100'],check=True,capture_output=True,text=True)
            spatial_report=json.loads((spatial_preview/'report.json').read_text());self.assertEqual(spatial_report['transfer_check']['other_photos_evaluated'],1)
            spatial_fit=json.loads((spatial_preview/'scene-01'/'fit.json').read_text());self.assertEqual(spatial_fit['fitted_recipe']['kind'],'radial-rgb-v1')
            spatial_replay=root/'spatial-replay';subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(spatial_preview),'--out',str(spatial_replay)],check=True,capture_output=True,text=True)
            with Image.open(spatial_preview/'scene-01'/'samples'/'image1.png') as first,Image.open(spatial_replay/'scene-01'/'exports'/'image1.png') as replayed:np.testing.assert_array_equal(np.asarray(first),np.asarray(replayed))
            gaussian=root/'gaussian-preview'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','grid','--solver','least-squares','--regularization','.0001','--target-engine','gaussian-soft','--prompt','warm and lift shadows','--target-local-light','.3','--target-light-center','.2','.3','--target-light-radius','.18','.23','--preview-only','--preview-samples','2','--evaluate-transfer','--full-res-reference','--out',str(gaussian),'--size','100'],check=True,capture_output=True,text=True)
            gaussian_fit=json.loads((gaussian/'scene-01'/'fit.json').read_text());self.assertEqual(gaussian_fit['target_light_center'],[.2,.3]);self.assertEqual(gaussian_fit['target_light_radius'],[.18,.23]);self.assertEqual(gaussian_fit['target_local_light'],.3)
            self.assertEqual(json.loads((gaussian/'report.json').read_text())['transfer_check']['other_photos_evaluated'],1);self.assertTrue((gaussian/'scene-01'/'target-full.jpg').exists())
            invalid=root/'bad-gaussian';p=subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','grid','--solver','least-squares','--target-engine','gaussian-soft','--target-light-radius','0','.2','--out',str(invalid)],capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertFalse(invalid.exists())
            local_prompt=root/'local-prompt'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','grid','--solver','least-squares','--regularization','.0001','--target-engine','gaussian-soft','--prompt','warm and slightly brighten upper left','--preview-only','--out',str(local_prompt),'--size','100'],check=True,capture_output=True,text=True)
            local_fit=json.loads((local_prompt/'scene-01'/'fit.json').read_text());self.assertEqual(local_fit['target_light_center'],[.2,.2]);self.assertAlmostEqual(local_fit['target_local_light'],.15);self.assertEqual(local_fit['target_settings']['tone_temperature'],.12)
            conflict=root/'conflicting-prompt';p=subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','grid','--solver','least-squares','--target-engine','gaussian-soft','--prompt','brighten upper left','--target-local-light','.3','--out',str(conflict)],capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertFalse(conflict.exists())
            wrong_engine=root/'wrong-prompt-engine';p=subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','grid','--solver','least-squares','--target-engine','pillow-soft','--prompt','brighten upper left','--out',str(wrong_engine)],capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertFalse(wrong_engine.exists())
            grid_preview=root/'grid-preview'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groupfile),'--renderer','grid','--solver','least-squares','--regularization','.0001','--target-engine','spatial-soft','--prompt','warm and lift shadows','--preview-only','--preview-samples','2','--evaluate-transfer','--out',str(grid_preview),'--size','100'],check=True,capture_output=True,text=True)
            grid_fit=json.loads((grid_preview/'scene-01'/'fit.json').read_text());self.assertEqual(grid_fit['fitted_recipe']['kind'],'grid-rgb-v1')
            grid_review=json.loads((grid_preview/'review.json').read_text());self.assertIn('grid_out_of_gamut_fraction',grid_review['photos'][0]['clipping'])
            grid_replay=root/'grid-replay';subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(grid_preview),'--out',str(grid_replay)],check=True,capture_output=True,text=True)
            with Image.open(grid_preview/'scene-01'/'samples'/'image1.png') as first,Image.open(grid_replay/'scene-01'/'exports'/'image1.png') as replayed:np.testing.assert_array_equal(np.asarray(first),np.asarray(replayed))
            replay_review=json.loads((grid_replay/'review.json').read_text());self.assertIn('grid_out_of_gamut_fraction',replay_review['photos'][0]['clipping'])
            local_review=root/'local-review';local_script=Path(morph.__file__).with_name('review_local_light.py')
            subprocess.run([sys.executable,str(local_script),'build',str(groupfile),str(grid_preview),'--out',str(local_review),'--samples','2'],check=True,capture_output=True,text=True)
            self.assertTrue((local_review/'comparison-01.jpg').exists());self.assertEqual(json.loads((local_review/'index.json').read_text())['photos'],2)
            decision=local_review/'decisions.csv';decision.write_text('scene,local_strength\nscene-01,0.5\n')
            selected=root/'local-selections.json';subprocess.run([sys.executable,str(local_script),'compile',str(decision),'--out',str(selected)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(selected.read_text()),{'local_strengths':{'scene-01':.5}})
            selected_out=root/'selected-out';subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(grid_preview),'--out',str(selected_out),'--corrections',str(selected)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads((selected_out/'scene-01'/'applied.json').read_text())['local_strength'],.5)
            decision.write_text('scene,local_strength\nscene-01,2\n');bad=root/'bad-selections.json'
            rejected=subprocess.run([sys.executable,str(local_script),'compile',str(decision),'--out',str(bad)],capture_output=True,text=True);self.assertNotEqual(rejected.returncode,0);self.assertFalse(bad.exists())
            grid_local=root/'grid-local';subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(grid_preview),'--out',str(grid_local),'--strength','.7','--local-strength','0'],check=True,capture_output=True,text=True)
            local_record=json.loads((grid_local/'scene-01'/'applied.json').read_text());local_weights=grid_light.deserialize(local_record['applied_recipe']);fitted_weights=grid_light.deserialize(grid_fit['fitted_recipe'])
            np.testing.assert_allclose(local_weights[:10],fitted_weights[:10]*.7);np.testing.assert_array_equal(local_weights[10:],0);self.assertEqual(local_record['local_strength'],0)
            with Image.open(grid_local/'scene-01'/'exports'/'image1.png') as im:np.testing.assert_allclose(np.asarray(im,dtype=np.float32)/255,grid_light.render(morph.load(images[1]),local_weights),atol=1/255)
            rejected=root/'bad-local';bad=subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(grid_preview),'--out',str(rejected),'--local-strength','1.5'],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);self.assertFalse(rejected.exists())
            nongrid=root/'nongrid-local';bad=subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(spatial_preview),'--out',str(nongrid),'--local-strength','0'],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);self.assertFalse(nongrid.exists())
            self.assertFalse((sampled/'scene-01'/'exports').exists())
            with Image.open(sampled/'scene-01'/'samples'/'image1.png') as im:self.assertEqual(im.size,(100,80))
    def test_bounds_and_shape(self):
        with self.assertRaises(ValueError):morph.vec({'exposure':2})
        with self.assertRaises(ValueError):morph.soft_settings('warm',{'tone_exposure':.6})
        with self.assertRaises(ValueError):morph.fit(np.zeros((3,2,3)),np.zeros((2,2,3)))
        x=np.random.default_rng(2).random((10,12,3),dtype=np.float32)
        np.testing.assert_array_equal(morph.render(x,np.zeros(len(morph.NAMES))),x)
        self.assertFalse(morph.clipping_diagnostic(x,np.zeros(len(morph.NAMES)))['flag'])
        bright=np.full((20,20,3),.8,dtype=np.float32)
        self.assertTrue(morph.clipping_diagnostic(bright,morph.vec({'exposure':1}))['flag'])

    def test_replay_detects_changed_source_and_group_membership(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);photos=[]
            for i in range(2):
                p=root/f'{i}.png';Image.new('RGB',(24,24),(80+i*20,100,120)).save(p);photos.append(str(p))
            groups=root/'groups.json';group={'groups':[{'id':'scene-01','representative':photos[0],'photos':photos}]};groups.write_text(json.dumps(group));fit=root/'fit'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groups),'--preview-only','--out',str(fit),'--size','24'],check=True,capture_output=True,text=True)
            saved=Path(photos[1]).read_bytes();Image.new('RGB',(24,24),(10,20,30)).save(photos[1])
            command=[sys.executable,str(Path(morph.__file__)),'apply',str(groups),str(fit),'--out',str(root/'replay')]
            bad=subprocess.run(command,capture_output=True,text=True);self.assertNotEqual(bad.returncode,0);self.assertIn('file changed',bad.stderr);self.assertFalse((root/'replay').exists())
            subprocess.run(command+['--allow-changed-inputs'],check=True,capture_output=True,text=True)
            applied=json.loads((root/'replay'/'scene-01'/'applied.json').read_text());self.assertEqual(applied['source_validation'],'changed-inputs-allowed')
            Path(photos[1]).write_bytes(saved);group['groups'][0]['photos']=[photos[0]];groups.write_text(json.dumps(group))
            bad=subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groups),str(fit),'--out',str(root/'changed-group')],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);self.assertIn('scene membership changed',bad.stderr)

    def test_groups_must_not_duplicate_or_escape(self):
        with tempfile.TemporaryDirectory() as td:
            image=Path(td)/'a.jpg';image.touch()
            group={'id':'scene-01','representative':str(image),'photos':[str(image)]}
            self.assertEqual(len(morph.validate_groups({'groups':[group]})),1)
            with self.assertRaises(ValueError): morph.validate_groups({'groups':[group,dict(group,id='scene-02')]})
            with self.assertRaises(ValueError): morph.validate_groups({'groups':[dict(group,id='../outside')]})
            with self.assertRaises(ValueError): morph.validate_groups({'groups':[dict(group,representative='missing')]})

    def test_chronological_groups_split_on_time_and_preserve_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            paths=[]
            for i,stamp in enumerate(['2026:04:24 12:00:00','2026:04:24 12:01:00','2026:04:24 12:20:00',None]):
                path=Path(td)/f'{i}.jpg';exif=Image.Exif()
                if stamp:exif[306]=stamp
                Image.new('RGB',(20,20),(i*20,50,90)).save(path,exif=exif);paths.append(path)
            groups=morph.chronological_groups(paths,5)
            self.assertEqual([len(g['photos']) for g in groups],[2,1,1])
            self.assertEqual(len(morph.validate_groups({'groups':groups})),3)
            hybrid=morph.hybrid_groups(paths,5,2)
            self.assertEqual(len({p for g in hybrid for p in g['photos']}),4)
            self.assertEqual(len(morph.validate_groups({'groups':hybrid})),3)

    def test_export_preserves_capture_metadata_and_normalizes_orientation(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source.jpg';output=Path(td)/'out.jpg';exif=Image.Exif();exif[274]=6;exif[306]='2026:04:24 12:00:00'
            Image.new('RGB',(20,30),(30,90,150)).save(source,exif=exif)
            pixels=morph.load(source);morph.save_render(pixels,np.zeros(len(morph.NAMES)),output,source=source)
            with Image.open(output) as im:
                self.assertEqual(im.size,(30,20));self.assertEqual(im.getexif().get(274),1);self.assertEqual(im.getexif().get(306),'2026:04:24 12:00:00')

    def test_target_and_album_refuse_to_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'scene-01'/'exports'/'source.jpg';source.parent.mkdir(parents=True)
            Image.new('RGB',(20,20),(30,90,150)).save(source);original_bytes=source.read_bytes()
            target=subprocess.run([sys.executable,str(Path(morph.__file__)),'target',str(source),'--out',str(source)],capture_output=True,text=True)
            self.assertNotEqual(target.returncode,0);self.assertIn('overwrite an input',target.stderr);self.assertEqual(source.read_bytes(),original_bytes)
            groups=root/'groups.json';groups.write_text(json.dumps({'groups':[{'id':'scene-01','representative':str(source),'photos':[str(source)]}]}))
            album=subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groups),'--out',str(root)],capture_output=True,text=True)
            self.assertNotEqual(album.returncode,0);self.assertIn('overwrite an input',album.stderr);self.assertEqual(source.read_bytes(),original_bytes)
            album=subprocess.run([sys.executable,str(Path(morph.__file__)),'album',str(groups),'--out',str(root),'--overwrite'],capture_output=True,text=True)
            self.assertNotEqual(album.returncode,0);self.assertEqual(source.read_bytes(),original_bytes)
            other=root/'existing.jpg';other.write_bytes(b'keep')
            command=[sys.executable,str(Path(morph.__file__)),'target',str(source),'--out',str(other)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertEqual(other.read_bytes(),b'keep')
            subprocess.run(command+['--overwrite'],check=True,capture_output=True,text=True)
            with Image.open(other) as result:self.assertEqual(result.size,(20,20))
            originals=root/'reconstructions';originals.mkdir();original=originals/'test.jpg';Image.new('RGB',(20,20)).save(original);original_bytes=original.read_bytes()
            edits=root/'edits';edits.mkdir();Image.new('RGB',(20,20)).save(edits/'test.jpg')
            evaluation=subprocess.run([sys.executable,str(Path(morph.__file__)),'evaluate',str(originals),str(edits),'--out',str(root),'--overwrite'],capture_output=True,text=True)
            self.assertNotEqual(evaluation.returncode,0);self.assertEqual(original.read_bytes(),original_bytes)


    def test_gauss_newton_recovers_known_adjustments(self):
        rng=np.random.default_rng(5);original=rng.random((70,90,3),dtype=np.float32)*.7+.1
        expected=morph.vec({'exposure':.1,'temperature':.05,'contrast':.1,'shadows':.12,'highlights':-.08,'saturation':.09})
        target=morph.render(original,expected);fitted,metrics=morph.run_fit(original,target,method='gauss-newton')
        self.assertLess(metrics['rmse'],.001)
        self.assertLess(float(np.max(abs(fitted-expected))),.02)

    def test_selective_renderer_reconstructs_and_preserves_geometry(self):
        rng=np.random.default_rng(19);original=rng.random((65,75,3),dtype=np.float32)*.7+.1
        p=np.array([.1,.05,0,.05,.1,-.08,.03,.25,-.2]);target=selective.render(original,p)
        fitted,metrics=selective.fit(original,target)
        self.assertEqual(target.shape,original.shape)
        self.assertLess(metrics['rmse'],.001)
        other=rng.random((50,55,3),dtype=np.float32)*.7+.1
        self.assertLess(float(np.sqrt(np.mean((selective.render(other,p)-selective.render(other,fitted))**2))),.002)
        regularized,penalized=selective.fit(original,target,regularization=.1)
        self.assertLess(np.linalg.norm(regularized[7:]),np.linalg.norm(fitted[7:]))
        self.assertGreaterEqual(penalized['objective'],penalized['fit_mse'])

    def test_gamut_renderer_identity_and_known_recipe(self):
        rng=np.random.default_rng(8);x=rng.random((70,80,3),dtype=np.float32)*.8+.1
        np.testing.assert_allclose(gamut_renderer.render(x,np.zeros(9)),x,atol=1e-6)
        expected=gamut_renderer.vec({'exposure':.12,'temperature':.05,'contrast':.09,'shadows':.08,'highlights':-.1,'cool_saturation':-.15,'warm_saturation':.1})
        target=gamut_renderer.render(x,expected);fitted,metrics=gamut_renderer.fit(x,target)
        self.assertGreaterEqual(float(target.min()),0);self.assertLessEqual(float(target.max()),1)
        self.assertLess(metrics['rmse'],.001);self.assertLess(float(np.max(abs(fitted-expected))),.02)

    def test_polynomial_color_mapping_roundtrip(self):
        rng=np.random.default_rng(30);x=rng.random((50,70,3),dtype=np.float32)*.7+.1
        weights=poly_color.identity();weights[3,0]=.08;weights[1,2]=-.04
        target=poly_color.render(x,weights);fitted,metrics=poly_color.fit(x,target,regularization=1e-6)
        self.assertLess(metrics['rmse'],.001)
        np.testing.assert_allclose(poly_color.deserialize(poly_color.serialize(fitted)),fitted)
        np.testing.assert_allclose(poly_color.render(x,poly_color.identity()),x)
        self.assertEqual(poly_color.out_of_gamut(x,poly_color.identity()),0)
        extreme=poly_color.identity();extreme[-1,:]=1
        self.assertEqual(poly_color.out_of_gamut(x,extreme),1)

    def test_spatial_tiles_match_whole_image(self):
        x=np.random.default_rng(74).random((650,40,3),dtype=np.float32)*.8+.1;w=np.random.default_rng(3).normal(0,.03,(13,3));full=spatial_light.render(x,w)
        tiled=np.concatenate([spatial_light.render_tile(x[s:s+256],w,s,x.shape) for s in range(0,len(x),256)],axis=0)
        np.testing.assert_allclose(tiled,full,atol=1e-7)
        settings={'tone_exposure':.1,'tone_temperature':.04};target=spatial_light.target(x,settings,.3)
        parts=np.concatenate([spatial_light.target_tile(x[s:s+256],settings,.3,s,x.shape) for s in range(0,len(x),256)],axis=0)
        np.testing.assert_allclose(parts,target,atol=1e-7)
        target=spatial_light.gaussian_target(x,settings,.3,(.2,.3),(.18,.23))
        parts=np.concatenate([spatial_light.gaussian_target(x[s:s+256],settings,.3,(.2,.3),(.18,.23),s,x.shape) for s in range(0,len(x),256)],axis=0)
        np.testing.assert_allclose(parts,target,atol=1e-7)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source.png';Image.fromarray(np.uint8(x*255+.5)).save(source)
            groups=root/'groups.json';groups.write_text(json.dumps({'groups':[{'id':'scene-01','representative':str(source),'photos':[str(source)]}]}))
            fit=root/'fits'/'scene-01';fit.mkdir(parents=True);(fit/'fit.json').write_text(json.dumps({'renderer':'spatial','representative':str(source),'fitted_recipe':spatial_light.serialize(w)}))
            corrections=root/'corrections.json';corrections.write_text(json.dumps({'album':{'temperature':.04},'photos':{'scene-01':{'source.png':{'exposure':.12}}}}))
            out=root/'out';subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groups),str(root/'fits'),'--corrections',str(corrections),'--out',str(out)],check=True,capture_output=True,text=True)
            expected=gamut_renderer.render(morph.load(source),gamut_renderer.vec({'exposure':.12}));expected=spatial_light.render(expected,w);expected=gamut_renderer.render(expected,gamut_renderer.vec({'temperature':.04}))
            with Image.open(out/'scene-01'/'exports'/'source.png') as actual:np.testing.assert_allclose(np.asarray(actual,dtype=np.float32)/255,expected,atol=1/255)

    def test_grid_fit_and_tiled_render(self):
        x=np.random.default_rng(74).random((650,40,3),dtype=np.float32)*.8+.1
        w=np.random.default_rng(3).normal(0,.03,(37,3))
        full=grid_light.render(x,w)
        reference=np.clip(x+grid_light.features(x)@w,0,1)
        np.testing.assert_allclose(full,reference,atol=1e-7)
        tiled=np.concatenate([grid_light.render_tile(x[s:s+256],w,s,x.shape) for s in range(0,len(x),256)],axis=0)
        np.testing.assert_allclose(tiled,full,atol=1e-7)
        fitted,metrics=grid_light.fit_with_metrics(x,full,regularization=.0001)
        self.assertLess(metrics['fit_mse'],metrics['initial_mse'])
        np.testing.assert_array_equal(grid_light.deserialize(grid_light.serialize(fitted)),fitted)
        extreme=grid_light.identity();extreme[9,:]=1
        self.assertEqual(grid_light.out_of_gamut(x,extreme),1)
        self.assertEqual(grid_light.out_of_gamut(x,grid_light.identity()),0)
        tall=np.random.default_rng(1).random((1025,17,3),dtype=np.float32)
        raw=grid_light.render_raw(tall,w);expected=float(np.mean(np.any((raw[::2,::2]<0)|(raw[::2,::2]>1),axis=-1)))
        self.assertEqual(grid_light.out_of_gamut(tall,w),expected)
        pre=lambda sample:np.clip(sample*.8,0,1)
        raw=grid_light.render_raw(pre(tall),w);expected=float(np.mean(np.any((raw[::2,::2]<0)|(raw[::2,::2]>1),axis=-1)))
        self.assertEqual(grid_light.out_of_gamut(tall,w,pre),expected)
        with self.assertRaises(ValueError):grid_light.deserialize({'kind':'grid-rgb-v1','weights':[[float('nan')]]})

    def test_per_scene_local_strength_and_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);groups=[];weights=np.zeros((37,3));weights[0,0]=.1;weights[10,0]=.2
            for i in (1,2):
                photo=root/f'photo{i}.png';Image.new('RGB',(12,10),(100,120,140)).save(photo)
                scene=f'scene-0{i}';groups.append({'id':scene,'representative':str(photo),'photos':[str(photo)]})
                fit=root/'fits'/scene;fit.mkdir(parents=True);(fit/'fit.json').write_text(json.dumps({'renderer':'grid','representative':str(photo),'fitted_recipe':grid_light.serialize(weights)}))
            groupfile=root/'groups.json';groupfile.write_text(json.dumps({'groups':groups}))
            config=root/'corrections.json';config.write_text(json.dumps({'local_strengths':{'scene-01':0.,'scene-02':.8}}))
            out=root/'out';subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(root/'fits'),'--out',str(out),'--local-strength','.5','--corrections',str(config)],check=True,capture_output=True,text=True)
            for scene,value in [('scene-01',0.),('scene-02',.8)]:
                a=json.loads((out/scene/'applied.json').read_text());actual=grid_light.deserialize(a['applied_recipe']);self.assertEqual(a['local_strength'],value)
                np.testing.assert_allclose(actual[:10],weights[:10]);np.testing.assert_allclose(actual[10:],weights[10:]*value)
            for index,value in enumerate((True,-.1,1.1,'0.5',None)):
                config.write_text(json.dumps({'local_strengths':{'scene-02':value}}));bad=root/f'bad{index}'
                result=subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(root/'fits'),'--out',str(bad),'--corrections',str(config)],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())
            config.write_text(json.dumps({'local_strengths':{'missing':.2}}));bad=root/'unknown'
            result=subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(root/'fits'),'--out',str(bad),'--corrections',str(config)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(bad.exists())

    def test_solver_can_anchor_toward_a_shared_recipe(self):
        x=np.random.default_rng(47).random((50,60,3),dtype=np.float32)*.8+.1
        wanted=gamut_renderer.vec({'exposure':.25,'temperature':.1});target=gamut_renderer.render(x,wanted)
        anchor=gamut_renderer.vec({'exposure':-.1})
        unanchored,_=gamut_renderer.fit(x,target)
        anchored,metrics=solver.fit_gauss_newton(x,target,renderer=gamut_renderer.render,bounds=gamut_renderer.BOUNDS,penalty=np.ones(9)*.05,prior=anchor)
        self.assertLess(np.linalg.norm(anchored-anchor),np.linalg.norm(unanchored-anchor))
        self.assertGreaterEqual(metrics['objective'],metrics['fit_mse'])

    def test_prompt_grammar_rejects_unsupported_and_conflicting_edits(self):
        settings,parsed=prompt_target.parse('warm, slightly soft contrast and lift shadows')
        self.assertEqual(settings,{'tone_temperature':.12,'tone_contrast':-.1,'tone_shadows':.2})
        self.assertEqual(len(parsed),3)
        with self.assertRaises(ValueError):prompt_target.parse('warm, remove person')
        with self.assertRaises(ValueError):prompt_target.parse('warm and cool')
        for alias,canonical in prompt_target.ALIASES.items():
            self.assertEqual(prompt_target.parse(alias)[0],prompt_target.parse(canonical)[0])
            self.assertEqual(prompt_target.parse('slightly '+alias)[0],prompt_target.parse('slightly '+canonical)[0])
        with self.assertRaises(ValueError):prompt_target.parse('warm and cooler')
        with self.assertRaises(ValueError):prompt_target.parse('brighten and increase exposure')
        with self.assertRaises(ValueError):prompt_target.parse('do not brighten')
        settings,parsed,local=prompt_target.parse_with_local('warm and strongly darken lower right')
        self.assertEqual(settings,{'tone_temperature':.12});self.assertEqual(local['center'],[.8,.8]);self.assertAlmostEqual(local['strength'],-.45)
        with self.assertRaises(ValueError):prompt_target.parse_with_local('brighten upper left and darken center')
        with self.assertRaises(ValueError):prompt_target.parse_with_local('brighten the face')

    def test_warm_prompt_direction_survives_color_mapping_transfer(self):
        rng=np.random.default_rng(81);representative=rng.random((70,80,3),dtype=np.float32)*.7+.15;held=rng.random((70,80,3),dtype=np.float32)*.7+.15
        settings,_=prompt_target.parse('warm');target=morph.pillow_soft_target(representative,'neutral',settings)
        mapping,_=poly_color.fit(representative,target,regularization=.001);out=poly_color.render(held,mapping);neutral=morph.pillow_soft_target(held,'neutral',{})
        self.assertGreater(float(np.mean(out[...,0]-out[...,2])),float(np.mean(neutral[...,0]-neutral[...,2])))

    def test_exposure_suggestions_require_comparable_metadata(self):
        stamp=datetime(2026,1,1,12,0);ref={'camera':'A','timestamp':stamp,'proxy':1.}
        candidate={'camera':'A','timestamp':stamp+timedelta(minutes=1),'proxy':2.}
        result=suggest_exposure.propose(ref,candidate);self.assertTrue(result['eligible']);self.assertAlmostEqual(result['proposed_exposure'],-.25)
        self.assertFalse(suggest_exposure.propose(ref,dict(candidate,camera='B'))['eligible'])
        self.assertFalse(suggest_exposure.propose(ref,dict(candidate,timestamp=stamp+timedelta(hours=1)))['eligible'])
        self.assertFalse(suggest_exposure.propose(ref,None)['eligible'])

    def test_exposure_suggestion_and_sample_preview(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);paths=[]
            for i,shutter in enumerate((.01,.02,None)):
                p=root/f'{i}.jpg';ex=Image.Exif();ex[272]='CAMERA'
                if shutter is not None:ex[34665]={36867:'2026:01:01 12:00:00',33434:shutter,33437:2.8,34855:100}
                Image.new('RGB',(40,40),(100,120,140)).save(p,exif=ex);paths.append(str(p))
            group={'groups':[{'id':'scene-01','representative':paths[0],'photos':paths}]};groupfile=root/'groups.json';groupfile.write_text(json.dumps(group))
            fits=root/'fits'/'scene-01';fits.mkdir(parents=True);(fits/'fit.json').write_text(json.dumps({'renderer':'gamut','representative':paths[0],'fitted_recipe':gamut_renderer.recipe(np.zeros(9))}))
            corrections=root/'suggestions.json'
            subprocess.run([sys.executable,str(Path(suggest_exposure.__file__)),str(groupfile),str(root/'fits'),'--out',str(corrections)],check=True,capture_output=True,text=True)
            result=json.loads(corrections.read_text());self.assertAlmostEqual(result['photos']['scene-01']['1.jpg']['exposure'],-.25);self.assertNotIn('2.jpg',result['photos']['scene-01'])
            review=root/'review';subprocess.run([sys.executable,str(Path(review_exposure.__file__)),'build',str(groupfile),str(root/'fits'),str(corrections)+'.audit.json','--out',str(review),'--size','40'],check=True,capture_output=True,text=True)
            with open(review/'decisions.csv',newline='') as f:items=list(csv.DictReader(f))
            self.assertEqual(len(items),1);self.assertEqual(items[0]['decision'],'')
            empty=root/'unreviewed.json';subprocess.run([sys.executable,str(Path(review_exposure.__file__)),'compile',str(review/'decisions.csv'),'--out',str(empty)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(empty.read_text()),{'photos':{}})
            items[0]['decision']='accept'
            with open(review/'decisions.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=review_exposure.FIELDS);writer.writeheader();writer.writerows(items)
            approved=root/'approved.json';subprocess.run([sys.executable,str(Path(review_exposure.__file__)),'compile',str(review/'decisions.csv'),'--out',str(approved)],check=True,capture_output=True,text=True)
            self.assertEqual(json.loads(approved.read_text()),result)
            preview=root/'preview';subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(root/'fits'),'--corrections',str(corrections),'--preview-samples','2','--out',str(preview)],check=True,capture_output=True,text=True)
            report=json.loads((preview/'report.json').read_text());self.assertEqual(report['sample_count'],2);self.assertEqual(report['export_count'],0);self.assertTrue((preview/'scene-01'/'samples'/'1.jpg').exists())


if __name__=='__main__': unittest.main()
