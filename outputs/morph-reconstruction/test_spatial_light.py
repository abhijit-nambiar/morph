import unittest
import numpy as np
import poly_color,spatial_light

class SpatialTests(unittest.TestCase):
    def test_identity_preserves_pixels_and_dimensions(self):
        x=np.random.default_rng(20).random((60,80,3),dtype=np.float32)
        y=spatial_light.render(x,np.zeros((13,3)))
        np.testing.assert_allclose(x,y,atol=1e-7)
        self.assertEqual(x.shape,y.shape)
    def test_radial_target_transfers_to_different_image(self):
        rng=np.random.default_rng(21);x=rng.random((100,120,3),dtype=np.float32)*.75+.1;other=rng.random((90,110,3),dtype=np.float32)*.75+.1
        settings={'tone_exposure':.12,'tone_temperature':.04};target=spatial_light.target(x,settings,.3);truth=spatial_light.target(other,settings,.3)
        global_fit,_=poly_color.fit(x,target,regularization=.0001);local_fit=spatial_light.fit(x,target,regularization=.0001)
        err=lambda y:float(np.sqrt(np.mean((y-truth)**2)))
        self.assertLess(err(spatial_light.render(other,local_fit)),err(poly_color.render(other,global_fit)))
if __name__=='__main__':unittest.main()
