# pip install pyimkernel
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
        print('test_apply_filter_on_gray_img Passed')
        self.assertEqual(type(imkernel.apply_filter_on_gray_img(X=X_train[0], kernel_name='blur')), np.ndarray)
        self.assertEqual(type(imkernel.apply_filter_on_gray_img(X=X_train[0], kernel_name='all')), dict)
        self.assertEqual(type(imkernel.apply_filter_on_gray_img(X=X_train[0], kernel_name=['blur', 'Laplacian'])), dict)
        self.assertRaises(KeyError, imkernel.apply_filter_on_gray_img, X_train[0], 'blure')
        self.assertRaises(AttributeError, imkernel.apply_filter_on_gray_img, X_train[0], [1, 2])
        self.assertRaises(ValueError, imkernel.apply_filter_on_gray_img, X_train[0], ['blure', 'Laplasian'])
        self.assertRaises(ValueError, imkernel.apply_filter_on_gray_img, X_train[0], [])
        self.assertRaises(ValueError, imkernel.apply_filter_on_gray_img, np.array([1, 2, 3]), 'blur')
        self.assertRaises(TypeError, imkernel.apply_filter_on_gray_img, X_train[0], ('blur',))
        self.assertRaises(TypeError, imkernel.apply_filter_on_gray_img, X_train[0], tuple([1, 2, 3]))
        self.assertRaises(TypeError, imkernel.apply_filter_on_gray_img, ['blur'], 'blur')


    def test_apply_filter_on_color_img(self): # Test method 2
        print('test_apply_filter_on_color_img Passed')
        self.assertRaises(IndexError, imkernel.apply_filter_on_color_img, X_train[0], 'blur', False)
        self.assertRaises(IndexError, imkernel.apply_filter_on_color_img, np.array([1, 2, 3]), 'blur', False)
        self.assertRaises(ValueError, imkernel.apply_filter_on_color_img, X_train, 'blur', False)
        self.assertRaises(KeyError, imkernel.apply_filter_on_color_img, flower_img, 'blure', False)
        self.assertRaises(AttributeError, imkernel.apply_filter_on_color_img, flower_img, [1, 2], False)
        self.assertRaises(ValueError, imkernel.apply_filter_on_color_img, flower_img, ['blure', 'Laplasian'], False)
        self.assertRaises(ValueError, imkernel.apply_filter_on_color_img, flower_img, [], False)
        self.assertRaises(TypeError, imkernel.apply_filter_on_color_img, flower_img, ('blur',), False)
        self.assertRaises(TypeError, imkernel.apply_filter_on_color_img, flower_img, tuple([1, 2, 3]), False)
        self.assertRaises(TypeError, imkernel.apply_filter_on_color_img, ['blur'], 'blur', False)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(flower_img, with_resize=False)), dict)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(X=flower_img, kernel_name=['blur', 'Laplacian'], with_resize=False)), dict)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(flower_img, kernel_name='laplacian', with_resize=False)), np.ndarray)
        ##
        self.assertRaises(IndexError, imkernel.apply_filter_on_color_img, X_train[0], 'blur', True, 'auto')
        self.assertRaises(IndexError, imkernel.apply_filter_on_color_img, np.array([1, 2, 3]), 'blur', True, 'auto')
        self.assertRaises(ValueError, imkernel.apply_filter_on_color_img, X_train, 'blur', True, 'auto')
        self.assertRaises(KeyError, imkernel.apply_filter_on_color_img, flower_img, 'blure', True, 'auto')
        self.assertRaises(AttributeError, imkernel.apply_filter_on_color_img, flower_img, [1, 2], True, 'auto')
        self.assertRaises(ValueError, imkernel.apply_filter_on_color_img, flower_img, ['blure', 'Laplasian'], True, 'auto')
        self.assertRaises(ValueError, imkernel.apply_filter_on_color_img, flower_img, [], True, 'auto')
        self.assertRaises(TypeError, imkernel.apply_filter_on_color_img, flower_img, ('blur',), True, 'auto')
        self.assertRaises(TypeError, imkernel.apply_filter_on_color_img, flower_img, tuple([1, 2, 3]), True, 'auto')
        self.assertRaises(TypeError, imkernel.apply_filter_on_color_img, ['blur'], 'blur', True, 'auto')
        self.assertRaises(ValueError,  imkernel.apply_filter_on_color_img, flower_img, 'box blur', True, '200, 200')
        self.assertRaises(TypeError,  imkernel.apply_filter_on_color_img, flower_img, 'box blur', True, ('a', 'b'))
        self.assertRaises(ValueError,  imkernel.apply_filter_on_color_img, flower_img, 'box blur', True, (100, 100, 100))
        self.assertRaises(TypeError,  imkernel.apply_filter_on_color_img, flower_img, 'box blur', True, [100, 100, 100])
        self.assertEqual(type(imkernel.apply_filter_on_color_img(flower_img, with_resize=True)), dict)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(X=flower_img, kernel_name=['blur', 'Laplacian'], with_resize=True)), dict)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(flower_img, kernel_name='laplacian', with_resize=True)), np.ndarray)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(flower_img, kernel_name=['blur', 'sharpen'], with_resize=True, dsize=(64, 64))), dict)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(flower_img, kernel_name='all', with_resize=True, dsize=(64, 64))), dict)
        self.assertEqual(type(imkernel.apply_filter_on_color_img(flower_img, kernel_name='blur', with_resize=True, dsize=(64, 64))), np.ndarray)
        

    def test_imshow(self): # Test method 3
        print('test_imshow Passed')
        # The Grayscale Image
        imkernel.imshow(image=imkernel.apply_filter_on_gray_img(X_train[0], kernel_name='blur'), cmap=plt.cm.gray)
        g_figure = plt.get_fignums()
        self.assertTrue(len(g_figure) > 0, msg="No image displayed")
        # The Color-scale Image
        imkernel.imshow(image=imkernel.apply_filter_on_color_img(flower_img, kernel_name='laplacian', with_resize=True), figsize=(7, 6), cmap=plt.cm.gray)
        c_figure = plt.get_fignums()
        self.assertTrue(len(c_figure) > 0, msg="No image displayed")


if __name__ == "__main__":
    unittest.main()
