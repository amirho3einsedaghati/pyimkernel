# (!)pip install pyimkernel
import unittest
from pyimkernel import ApplyKernels
import mnist
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


X_train, X_test, y_train, y_test = mnist.train_images(), mnist.test_images(), mnist.train_labels(), mnist.test_labels()
imkernel = ApplyKernels(random_seed=0)
flower_img = cv2.imread(os.path.join('Images', '1.jpg'))


class TestApplyKernels(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print('setUpClass Done\n')
        
    @classmethod
    def tearDownClass(self):
        print('tearDownClass Done')
        
    def setUp(self):
        print('setUp Done')
        
    def tearDown(self):
        print('tearDown Done\n')
        
    def test_apply_filter_on_gray_img(self): # Test method 1
        print('test_apply_filter_on_gray_img Done')
        self.assertEqual(type(imkernel.apply_filter_on_gray_img(X=X_train[0], kernel_name='blur')), np.ndarray)
        self.assertEqual(type(imkernel.apply_filter_on_gray_img(X=X_train[0], kernel_name='all')), dict)
        self.assertEqual(type(imkernel.apply_filter_on_gray_img(X=X_train[0], kernel_name=['blur', 'Laplacian'])), dict)
        self.assertRaises(KeyError, imkernel.apply_filter_on_gray_img, X_train[0], 'blure')
        self.assertRaises(AttributeError, imkernel.apply_filter_on_gray_img, X_train[0], [1, 2])
        self.assertRaises(ValueError, imkernel.apply_filter_on_gray_img, X_train[0], ['blure', 'Laplasian'])
        self.assertRaises(ValueError, imkernel.apply_filter_on_gray_img, X_train[0], [])
        self.assertRaises(ValueError, imkernel.apply_filter_on_gray_img, np.array([1, 2, 3]), 'blur')

    def test_apply_filter_on_color_img(self): # Test method 2
        print('test_apply_filter_on_color_img Done')
        self.assertRaises(IndexError, imkernel.apply_filter_on_color_img, X_train[0], 'blur')
        self.assertRaises(IndexError, imkernel.apply_filter_on_color_img, np.array([1, 2, 3]), 'blur')
        self.assertRaises(ValueError, imkernel.apply_filter_on_color_img, X_train, 'blur')
        self.assertRaises(KeyError, imkernel.apply_filter_on_color_img, flower_img, 'blure')
        self.assertRaises(AttributeError, imkernel.apply_filter_on_color_img, flower_img, [1, 2])
        self.assertRaises(ValueError, imkernel.apply_filter_on_color_img, flower_img, ['blure', 'Laplasian'])
        self.assertRaises(ValueError, imkernel.apply_filter_on_color_img, flower_img, [])
        self.assertEqual(type(imkernel.apply_filter_on_color_img(flower_img)), dict)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(X=flower_img, kernel_name=['blur', 'Laplacian'])), dict)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(flower_img, kernel_name='laplacian', with_resize=True)), np.ndarray)
        
    def test_imshow(self): # Test method 3
        print('test_imshow Done')
        # Grayscale Image
        imkernel.imshow(image=imkernel.apply_filter_on_gray_img(X_train[0], kernel_name='blur'), cmap=plt.cm.gray)
        g_figure = plt.get_fignums()
        self.assertTrue(len(g_figure) > 0, msg="No image displayed")
        # Color Scale Image
        imkernel.imshow(image=imkernel.apply_filter_on_color_img(flower_img, kernel_name='laplacian', with_resize=True), figsize=(7, 6), cmap=plt.cm.gray)
        c_figure = plt.get_fignums()
        self.assertTrue(len(c_figure) > 0, msg="No image displayed")


if __name__ == "__main__":
    unittest.main()
