import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import cv2


kernels = {
    'blur' : np.array([[0.0625, 0.125, 0.0625], # guassian blur
                       [0.125,  0.25,  0.125],
                       [0.0625, 0.125, 0.0625]]),
        
    'bottom sobel' : np.array([[-1, -2, -1],
                               [ 0,  0, 0],
                               [ 1,  2, 1]]),
        
    'emboss' : np.array([[-2, -1, 0],
                         [-1,  1, 1],
                         [ 0,  1, 2]]),
        
    'identity' : np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]),
        
    'left sobel' : np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]),
        
    'outline' : np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]]),
        
    'right sobel' : np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]),
        
    'sharpen' : np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]]),
        
    'top sobel' : np.array([[ 1,  2,  1],
                            [ 0,  0,  0],
                            [-1, -2, -1]]),
        
    'horizontal edge' : np.array([[1, 0, -1],
                                  [1, 0, -1],
                                  [1, 0, -1]]),
        
    'vertical edge' : np.array([[ 1,  1, 1],
                                [ 0,  0, 0],
                                [-1, -1, -1]]),
        
    'box blur' : np.array([[1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9]]),
        
    'laplacian' : np.array([[0,  1, 0],
                            [1, -4, 1],
                            [0,  1, 0]]),
        
    'prewitt horizontal edge' : np.array([[-1, -1, -1],
                                          [ 0,  0, 0],
                                          [ 1,  1, 1]]),
        
    'prewitt vertical edge' : np.array([[-1, 0, 1],
                                        [-1, 0, 1],
                                        [-1, 0, 1]]),
        
    'high-pass filter' : np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]]),
        
    'unsharp masking' : np.array([[-1, -1, -1],
                                  [ -1,  9, -1],
                                  [ -1, -1, -1]]),
        
    'dilate' : np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]]),
        
        
    'soften' : np.array([[1/16, 1/8, 1/16],
                         [1/8,  1/4, 1/8],
                         [1/16, 1/8, 1/16]]),
        
    'scharr horizontal edge' : np.array([[-3,  0, 3],
                                         [-10, 0, 10],
                                         [-3,  0, 3]]),
         
    'scharr vertical edge' : np.array([[-3, -10,  -3],
                                       [ 0,  0,    0],
                                       [ 3,  10,   3]]),

    'motion blur' : np.array([[1/9,  0,   0],
                              [0,    1/9, 0],
                              [0,    0,   1/9]])
    }


class ApplyKernels():
    """
        To reach a wide range of effects and enhancements in digital images, You can use this class in various image processing tasks.


        Attributes:
        - random_seed: int, default=42
            To get the same results each time you run the function, You should initialize it with your desired value.

        Methods:
        - apply_filter_on_gray_img: Apply some kernel(s) on a grayscale image.
        - apply_filter_on_color_img: Apply some kernel(s) on a color-scale image.
        - imshow: Show the input image.
    """
    def __init__(self, random_seed:int=42):
        self.random_seed = random_seed


    def __get_filter_image(self, X, kernel_name):
        filtered_image = np.zeros(X.shape)
        for i, j in product(range(X.shape[0]), range(X.shape[1])):
            if i <= 25 and j <= 25:
                filtered_image[i, j] = np.sum(X[i : i+3, j : j+3] * kernels[kernel_name])
            if i > 25 and j <= 25:
                filtered_image[i, j] = np.sum(X[(i + 1)-3 : i+1, j : j+3] * kernels[kernel_name])
            if j > 25 and i <= 25:
                filtered_image[i, j] = np.sum(X[i : i+3, (j + 1)-3 : j+1] * kernels[kernel_name])
            if i > 25 and j > 25:
                filtered_image[i, j] = np.sum(X[(i + 1)-3 : i+1, (j + 1)-3 : j+1] * kernels[kernel_name])
        return filtered_image


    def __implementation(self, X, kernel_name):
        """
        It's used for applying filter(s) on an image using the private method __get_filter_image. So, It returns a filtered image
        """
        if len(X.shape) == 2: 
            if type(kernel_name) == list:
                if len(kernel_name) != 0:
                    try:
                        k_name = kernel_name[0].lower()
                    except:
                        raise AttributeError(f'{type(kernel_name[0])} object has no attribute `lower`')
                    else:
                        del k_name
                        k_names = [k_name.lower() for k_name in kernel_name if k_name.lower() in kernels.keys()] 
                        if len(k_names) == 0:
                            raise ValueError('There is no valid kernel name. For more info, please read the Docstring :)')
                        elif len(k_names) == 1:
                            kernel_name = k_names[0]
                        else:
                            filtered_images = {}
                            for k_name in k_names:
                                filtered_image = self.__get_filter_image(X, k_name)
                                filtered_images[k_name] = filtered_image
                            return filtered_images # a dictionary of various filters of an image 
                else:
                    raise ValueError('There is no item in the kernel_name parameter')
                
            if type(kernel_name) == str and kernel_name.lower() == 'all':
                filtered_images = {}
                for k_name in kernels.keys():
                    filtered_image = self.__get_filter_image(X, k_name)
                    filtered_images[k_name] = filtered_image
                return filtered_images # a dictionary of various filters of an image 
            
            elif type(kernel_name) == str and kernel_name.lower() in kernels.keys():
                filtered_image = self.__get_filter_image(X, kernel_name)
                return filtered_image # a 2-D Array

            else:
                if type(kernel_name) == tuple:
                    raise TypeError(f'kernel_name must be a string or list, Not {type(kernel_name)}. For more info, Please read the Docstring :)')
                else:
                    raise KeyError(f'There is no kernel named {kernel_name}. For more info, please read the Docstring :)')
        else:
            return 0
        

    def apply_filter_on_gray_img(self, X:np.ndarray, kernel_name='all'):
        """
        Apply some kernel(s) on a grayscale image.
            
            
        Parameters:
        ------------
        - X: `numpy.ndarray` (a 2-D array) 
            The grayscale image on which the filter(s) will be apply.
            
        - kernel_name: str or list, default='all'
        The list of valid kernels:
        'blur' : The blur kernel applies a smoothing effect, averaging the pixel values in the neighborhood.
        'bottom sobel' : The bottom sobel kernel emphasizes edges in the bottom directions.
        'emboss' : The emboss kernel creates a 3D embossed effect in the image.
        'identity' : The identity kernel does not modify the image and leaves it unchanged.
        'left sobel' : The left sobel kernel emphasizes edges in the left directions.
        'outline' : The outline kernel detects edges and boundaries by emphasizing the differences in intensity between neighboring pixels.
        'right sobel' : The right sobel kernel emphasizes edges in the right directions.
        'sharpen' : The sharpen kernel enhances edges and details in an image. 
        'top sobel' : The top sobel kernel emphasizes edges in the top directions.
        'horizontal edge' : The horizontal edge kernel highlights horizontal edges.
        'vertical edge' : The vertical edge kernel highlights vertical edges.
        'box blur' : The box blur kernel is similar to the blur kernel. It applies a simple averaging operation to create a blur effect, but with equal weights for all neighboring pixels.
        'laplacian' : The Laplacian kernel is used for edge detection and image sharpening.
        'prewitt horizontal edge' : The prewitt horizontal edge kernel is similar to the bottom sobel kernel, emphasizing edges in the horizontal directions.
        'prewitt vertical edge' : The prewitt vertical edge kernel is similar to the right sobel kernel, emphasizing edges in the horizontal directions.
        'high-pass filter' : The high-pass filter kernel enhances the details and edges in an image while reducing the low-frequency components.
        'unsharp masking' : The unsharp masking kernel is used for image sharpening. It enhances the edges and details in an image by subtracting a blurred version of the image from the original.
        'dilate' : The dilate kernel is used in morphological operations such as dilation, which expands regions of bright pixels in an image.
        'soften' : The soften kernel is used to reduce image noise and create a smoother appearance while preserving overall image details.
        'scharr horizontal edge': The scharr horizontal edge kernel is used for edge detection and gradient estimation along the horizontal direction. It provide more weight to the central pixel and its immediate neighbors.	
        'scharr vertical edge': The scharr vertical edge kernel is used for edge detection and gradient estimation along the vertical direction. It provide more weight to the central pixel and its immediate neighbors.
        'motion blur' : The motion blur kernel is used to simulate the effect of motion in an image. It achieves this by applying a linear blur in a specific direction. The kernel consists of non-zero values along a line in the direction of motion, with zeros elsewhere. When convolved with the image, it creates streak-like blurs that mimic the appearance of objects in motion.
        

        Returns:
        ------------
        `numpy.ndarray`(a 2-D Array), Dict, or error message
        """
        if type(X) == np.ndarray:
            if type(kernel_name) == str or type(kernel_name) == list:
                np.random.seed(self.random_seed)  
                filtered_image = self.__implementation(X, kernel_name)
                if type(filtered_image) == int and filtered_image == 0:
                    raise ValueError(f'Expected 2 axes, but got {len(X.shape)}!')
                else:
                    return filtered_image
            else:
                raise TypeError(f'kernel_name must be a string or list, Not {type(kernel_name)}. For more info, Please read the Docstring :)')
        else:
            raise TypeError(f'X must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')


    def apply_filter_on_color_img(self, X, kernel_name='all', with_resize:bool=False):
        """
        Apply some kernel(s) on a color-scale image.
            
            
        Parameters:
        ------------
        - X: array-like for example (499, 635, 3)
            The color-scale image on which the filter(s) will be apply.
            
        - kernel_name: str or list, default='all'
        The list of valid kernels:
        'blur' : The blur kernel applies a smoothing effect, averaging the pixel values in the neighborhood.
        'bottom sobel' : The bottom sobel kernel emphasizes edges in the bottom directions.
        'emboss' : The emboss kernel creates a 3D embossed effect in the image.
        'identity' : The identity kernel does not modify the image and leaves it unchanged.
        'left sobel' : The left sobel kernel emphasizes edges in the left directions.
        'outline' : The outline kernel detects edges and boundaries by emphasizing the differences in intensity between neighboring pixels.
        'right sobel' : The right sobel kernel emphasizes edges in the right directions.
        'sharpen' : The sharpen kernel enhances edges and details in an image. 
        'top sobel' : The top sobel kernel emphasizes edges in the top directions.
        'horizontal edge' : The horizontal edge kernel highlights horizontal edges.
        'vertical edge' : The vertical edge kernel highlights vertical edges.
        'box blur' : The box blur kernel is similar to the blur kernel. It applies a simple averaging operation to create a blur effect, but with equal weights for all neighboring pixels.
        'laplacian' : The Laplacian kernel is used for edge detection and image sharpening.
        'prewitt horizontal edge' : The prewitt horizontal edge kernel is similar to the bottom sobel kernel, emphasizing edges in the horizontal directions.
        'prewitt vertical edge' : The prewitt vertical edge kernel is similar to the right sobel kernel, emphasizing edges in the horizontal directions.
        'high-pass filter' : The high-pass filter kernel enhances the details and edges in an image while reducing the low-frequency components.
        'unsharp masking' : The unsharp masking kernel is used for image sharpening. It enhances the edges and details in an image by subtracting a blurred version of the image from the original.
        'dilate' : The dilate kernel is used in morphological operations such as dilation, which expands regions of bright pixels in an image.
        'soften' : The soften kernel is used to reduce image noise and create a smoother appearance while preserving overall image details.
        'scharr horizontal edge': The scharr horizontal edge kernel is used for edge detection and gradient estimation along the horizontal direction. It provide more weight to the central pixel and its immediate neighbors.	
        'scharr vertical edge': The scharr vertical edge kernel is used for edge detection and gradient estimation along the vertical direction. It provide more weight to the central pixel and its immediate neighbors.
        'motion blur' : The motion blur kernel is used to simulate the effect of motion in an image. It achieves this by applying a linear blur in a specific direction. The kernel consists of non-zero values along a line in the direction of motion, with zeros elsewhere. When convolved with the image, it creates streak-like blurs that mimic the appearance of objects in motion.
        
        
        - with_resize: bool, default=False
            To improve the speed of rendering matrices, You can use this parameter.
            note: Assigning the True value to the with_resize parameter results in a grayscale image, but Assigning the False value to the with_resize parameter results in a color-scale image.
            1. if the number of rows and columns in the input image are bigger than 400 pixels, and you assign True to the with_resize parameter, the number of rows and columns will change to 400 pixels and the kernel(s) will be apply on this image.
            2. if the number of rows and columns in the input image are smaller than 400 pixels, and you assign True/False to the with_resize parameter, the number of rows and columns won't change. So, the kernel(s) will be apply on the input image.

        Returns:
        ------------
        `numpy.ndarray`(a 2-D Array), Dict, or error message
        """
        if type(X) == np.ndarray:
            if type(kernel_name) == str or type(kernel_name) == list:
                try: 
                    axis2_val = X.shape[2]
                except:
                    raise IndexError(f'Expected 3 axes but got {len(X.shape)}!')
                else:
                    del axis2_val
                    if X.shape[2] == 3 and len(X.shape) == 3:
                        np.random.seed(self.random_seed)  
                        X = X.reshape(X.shape[0], -1) # convert the input array to a 2-D array
                        if with_resize == True:
                            if X.shape[0] > 400 and X.shape[1] > 400:
                                X = cv2.resize(X, (400, 400))
                                filtered_image = self.__implementation(X, kernel_name)
                                return filtered_image # a grayscale image or a dictioary of grayscale images
                            else:
                                with_resize = False
                        if with_resize == False:
                            filtered_image = self.__implementation(X, kernel_name)
                            if type(filtered_image) == np.ndarray:
                                filtered_image.reshape(filtered_image.shape[0], filtered_image.shape[1] // 3, 3)
                            return filtered_image # a color-scale image or a dictioary of grayscale images
                    else:
                        raise ValueError(f'Expected 3 axes and 3 channels but got {len(X.shape)} axes and {X.shape[2]} channels!')
            else:
                raise TypeError(f'kernel_name must be a string or list, Not {type(kernel_name)}. For more info, Please read the Docstring :)')
        else:
            raise TypeError(f'X must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')
    

    def imshow(self, image, figsize:tuple=(5, 5), cmap=None):
        """
        Show the input image.

             
        Parameters:
        ------------
        - image: array-like or PIL image
            The image data
        
        - figsize: tuple, default=(10, 10)
            The figure size
        
        - cmap: str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            It maps scalar data to colors.
        """
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap=cmap)
        plt.tight_layout()

