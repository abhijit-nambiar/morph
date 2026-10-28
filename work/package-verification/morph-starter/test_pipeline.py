import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
from PIL import Image
import morph
import selective
import gamut_renderer
import poly_color

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
            poly_replay=root/'polynomial-replay'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(output9),'--out',str(poly_replay)],check=True,capture_output=True,text=True)
            with Image.open(output9/'scene-01'/'exports'/'image1.png') as first,Image.open(poly_replay/'scene-01'/'exports'/'image1.png') as replayed:np.testing.assert_array_equal(np.asarray(first),np.asarray(replayed))
            poly_corrections={'album':{'temperature':.04},'scenes':{'scene-01':{'contrast':.08}},'photos':{'scene-01':{'image1.png':{'exposure':.2}}}}
            poly_correction_file=root/'poly-corrections.json';poly_correction_file.write_text(json.dumps(poly_corrections));poly_corrected=root/'poly-corrected'
            subprocess.run([sys.executable,str(Path(morph.__file__)),'apply',str(groupfile),str(output9),'--out',str(poly_corrected),'--corrections',str(poly_correction_file)],check=True,capture_output=True,text=True)
            expected=gamut_renderer.render(morph.load(images[1]),gamut_renderer.vec({'exposure':.2}));expected=poly_color.render(expected,poly_color.deserialize(poly['fitted_recipe']));expected=gamut_renderer.render(expected,gamut_renderer.vec({'contrast':.08}));expected=gamut_renderer.render(expected,gamut_renderer.vec({'temperature':.04}))
            with Image.open(poly_corrected/'scene-01'/'exports'/'image1.png') as actual:np.testing.assert_allclose(np.asarray(actual,dtype=np.float32)/255,expected,atol=1/255)
            applied=json.loads((poly_corrected/'scene-01'/'applied.json').read_text());self.assertEqual(applied['correction_renderer'],'gamut');self.assertEqual(applied['renderer'],'polynomial')
    def test_bounds_and_shape(self):
        with self.assertRaises(ValueError):morph.vec({'exposure':2})
        with self.assertRaises(ValueError):morph.soft_settings('warm',{'tone_exposure':.6})
        with self.assertRaises(ValueError):morph.fit(np.zeros((3,2,3)),np.zeros((2,2,3)))
        x=np.random.default_rng(2).random((10,12,3),dtype=np.float32)
        np.testing.assert_array_equal(morph.render(x,np.zeros(len(morph.NAMES))),x)
        self.assertFalse(morph.clipping_diagnostic(x,np.zeros(len(morph.NAMES)))['flag'])
        bright=np.full((20,20,3),.8,dtype=np.float32)
        self.assertTrue(morph.clipping_diagnostic(bright,morph.vec({'exposure':1}))['flag'])

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


if __name__=='__main__': unittest.main()
