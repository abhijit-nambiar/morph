import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

try:
    import cv2
except ImportError:
    cv2=None


@unittest.skipIf(cv2 is None,'Optional OpenCV normalization dependency is unavailable')
class PhotometricNormalizationTest(unittest.TestCase):
    def test_matching_recovers_unknown_setting_change_and_abstains_without_overlap(self):
        import morph
        import gamut_renderer
        import photometric_normalization as normalization
        rng=np.random.default_rng(90)
        image=Image.new('RGB',(420,340),(95,115,130));draw=ImageDraw.Draw(image)
        for _ in range(120):
            x,y=rng.integers(0,380),rng.integers(0,300)
            color=tuple(int(v) for v in rng.integers(35,220,3))
            draw.rectangle((int(x),int(y),int(x+30),int(y+30)),fill=color)
        clean=np.asarray(image,dtype=np.float32)/255
        # Independent perturbation expression; its parameters are never supplied
        # to feature matching or correction estimation.
        changed=1/(1+np.exp(-(np.log(clean/(1-clean))*1.035+np.array([.13,.08,.02]))))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);anchor=root/'anchor.png';source=root/'source.png';blank=root/'blank.png'
            image.save(anchor);Image.fromarray(np.uint8(changed*255+.5)).save(source)
            Image.new('RGB',image.size,(100,100,100)).save(blank)
            features={str(p):normalization.extract(p,512) for p in (anchor,source,blank)}
            corrections,_=normalization.solve_scene([str(source),str(blank)],str(anchor),features)
            correction=corrections[str(source)]
            self.assertEqual(correction['status'],'estimated')
            restored=normalization.render(morph.load(source),correction)
            self.assertLess(np.mean((restored-clean)**2),np.mean((changed-clean)**2)*.08)
            self.assertEqual(corrections[str(blank)]['status'],'review-unconnected')
            recipe=normalization.gamut_recipe(correction)
            np.testing.assert_allclose(restored,gamut_renderer.render(morph.load(source),gamut_renderer.vec(recipe)),atol=2e-6)


if __name__=='__main__':unittest.main()
