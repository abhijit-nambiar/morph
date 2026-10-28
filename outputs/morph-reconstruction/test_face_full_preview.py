import unittest
import numpy as np
import morph,face_light_experiment as face,face_independent_experiment as independent,face_full_preview as preview

class FaceFullPreviewTests(unittest.TestCase):
    def test_tiled_outputs_match_whole_image_models(self):
        x=np.random.default_rng(1).random((650,40,3),dtype=np.float32)*.8+.1
        boxes=[{'x':.18,'y':.26,'width':.17,'height':.2},{'x':.6,'y':.69,'width':.15,'height':.13}]
        settings={'tone_exposure':.1,'tone_temperature':.04};w=np.random.default_rng(4).normal(0,.02,(13,3))
        expected=face.render(x,boxes,w);tiled=np.concatenate([preview.model_tile(x[i:i+256],w,boxes,i,x.shape) for i in range(0,len(x),256)],axis=0)
        np.testing.assert_allclose(tiled,expected,atol=1e-7)
        expected=independent.target(x,boxes,settings,.3);tiled=np.concatenate([preview.target_tile(x[i:i+256],boxes,settings,.3,i,x.shape) for i in range(0,len(x),256)],axis=0)
        np.testing.assert_allclose(tiled,expected,atol=1e-7)

if __name__=='__main__':unittest.main()
