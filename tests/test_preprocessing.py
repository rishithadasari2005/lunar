import unittest
from preprocessing import ImagePreprocessor
import numpy as np

class TestImagePreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        self.synthetic = self.preprocessor.create_synthetic_lunar_image(size=(64, 64), crater_density=0.2)

    def test_grayscale_conversion(self):
        gray = self.preprocessor._convert_to_grayscale(np.stack([self.synthetic]*3, axis=-1))
        self.assertEqual(gray.shape, self.synthetic.shape)
        self.assertTrue(np.all(gray >= 0) and np.all(gray <= 255))

    def test_resize(self):
        resized = self.preprocessor._resize_image(self.synthetic, (32, 32))
        self.assertEqual(resized.shape, (32, 32))

if __name__ == '__main__':
    unittest.main() 