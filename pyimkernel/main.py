import numpy as np
import matplotlib.pyplot as plt
import cv2



kernels = {
    'guassian blur 3x3' : (1 / 16.0) * np.array([[1, 2, 1], # or blur 3x3
                                                 [2, 4, 2],
                                                 [1, 2, 1]]),

    'guassian blur 5x5' : (1 / 256.0) * np.array([[1, 4,  6,  4,  1], # or blur 5x5
                                                  [4, 16, 24, 16, 4],
                                                  [6, 24, 36, 24, 6],
                                                  [4, 16, 24, 16, 4],
                                                  [1, 4,  6,  4,  1]]),      
    'bottom sobel 3x3' : np.array([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]]),
    
    'bottom sobel 5x5' : np.array([[-1, -1, -2, -1, -1],
                                   [-1, -1, -2, -1, -1],
                                   [ 0,  0,  0,  0,  0],
                                   [ 1,  1,  2,  1,  1],
                                   [ 1,  1,  2,  1,  1]]),
        
    'left sobel 3x3' : np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]]),

    'left sobel 5x5' : np.array([[1, 1, 0, -1, -1],
                                 [2, 2, 0, -2, -2],
                                 [3, 3, 0, -3, -3],
                                 [2, 2, 0, -2, -2],
                                 [1, 1, 0, -1, -1]]),
        
    'right sobel 3x3' : np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]),

    'right sobel 5x5' : np.array([[-1, -1, 0, 1, 1],
                                  [-2, -2, 0, 2, 2],
                                  [-3, -3, 0, 3, 3],
                                  [-2, -2, 0, 2, 2],
                                  [-1, -1, 0, 1, 1]]),

    'top sobel 3x3' : np.array([[ 1,  2,  1],
                                [ 0,  0,  0],
                                [-1, -2, -1]]),
    
    'top sobel 5x5' : np.array([[ 1,  1,  2,  1,  1],
                                [ 1,  1,  2,  1,  1],
                                [ 0,  0,  0,  0,  0],
                                [-1, -1, -2, -1, -1],
                                [-1, -1, -2, -1, -1]]),

    'emboss 3x3' : np.array([[-2, -1, 0],
                             [-1,  1, 1],
                             [ 0,  1, 2]]),
    
    'emboss 5x5' : np.array([[-2, -1,  0,  1,  2],
                             [-1, -1,  0,  1,  1],
                             [ 0,  0,  0,  0,  0],
                             [ 1,  1,  0, -1, -1],
                             [ 2,  1,  0, -1, -2]]),
        
    'identity 3x3' : np.array([[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]]),

    'identity 5x5' : np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]]),

    'outline 3x3' : np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]]),

    'outline 5x5' : np.array([[ -1, -1,  -1,  -1,  -1],
                               [-1,  2,   2,   2,  -1],
                               [-1,  2,   8,   2,  -1],
                               [-1,  2,   2,   2,  -1],
                               [-1, -1,  -1,  -1,  -1]]),
        
    'sharpen 3x3' : np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]]),
    
    'sharpen 5x5' : np.array([[-1, -1, -1, -1, -1],
                              [-1,  2,  2,  2, -1],
                              [-1,  2,  16, 2, -1],
                              [-1,  2,  2,  2, -1],
                              [-1, -1, -1, -1, -1]]),

    'vertical edge 3x3' : np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]]),

    'vertical edge 5x5' : np.array([[-1, -1, 0, 1, 1],
                                    [-1, -1, 0, 1, 1],
                                    [-1, -1, 0, 1, 1],
                                    [-1, -1, 0, 1, 1],
                                    [-1, -1, 0, 1, 1]]),
        
    'horizontal edge 3x3' : np.array([[ 1,  1, 1],
                                      [ 0,  0, 0],
                                      [-1, -1,-1]]),

    'horizontal edge 5x5' : np.array([[ 1,  1,  1,  1,  1],
                                      [ 1,  1,  1,  1,  1],
                                      [ 0,  0,  0,  0,  0],
                                      [-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]),
        
    'box blur 3x3' : (1 / 9.0) * np.array([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]]),
    
    'box blur 5x5' : (1 / 25.0) * np.array([[1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1]]),
    'laplacian 3x3' : np.array([[0,  1, 0],
                                [1, -4, 1],
                                [0,  1, 0]]),

    'laplacian 5x5' : np.array([[ 0,  0, -1,   0,  0],
                                [ 0, -1, -2,  -1,  0],
                                [-1, -2,  16, -2, -1],
                                [ 0, -1, -2,  -1,  0], 
                                [ 0,  0, -1,   0,  0]]),
        
    'prewitt horizontal edge 3x3' : np.array([[-1, -1, -1],
                                              [ 0,  0,  0],
                                              [ 1,  1,  1]]),

    'prewitt horizontal edge 5x5' : np.array([[-1, -1, -1, -1, -1],
                                              [-1, -1, -1, -1, -1],
                                              [0,   0,  0,  0,  0],
                                              [1,   1,  1,  1,  1],
                                              [1,   1,  1,  1,  1]]),
        
    'prewitt vertical edge 3x3' : np.array([[-1, 0, 1],
                                            [-1, 0, 1],
                                            [-1, 0, 1]]),

    'prewitt vertical edge 5x5' : np.array([[-1, -1, 0, 1, 1],
                                            [-1, -1, 0, 1, 1],
                                            [-1, -1, 0, 1, 1],
                                            [-1, -1, 0, 1, 1],
                                            [-1, -1, 0, 1, 1]]),
        
    'high-pass filter 3x3' : np.array([[-1, -1, -1],
                                       [-1,  8, -1],
                                       [-1, -1, -1]]),

    'high-pass filter 5x5' : np.array([[-1, -1, -1, -1, -1],
                                       [-1, -1, -1, -1, -1],
                                       [-1, -1, 24, -1, -1],
                                       [-1, -1, -1, -1, -1],
                                       [-1, -1, -1, -1, -1]]),

    'unsharp masking 3x3' : np.array([[ -1, -1, -1],
                                      [ -1,  9, -1],
                                      [ -1, -1, -1]]),

    'unsharp masking 5x5' : (-1 / 256.0) * np.array([[1, 4,   6,   4,  1],
                                                     [4, 16,  24,  16, 4],
                                                     [6, 24, -476, 24, 6],
                                                     [4, 16,  24,  16, 4],
                                                     [1, 4,   6,   4,  1]]),
    'dilation 3x3' : np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]]),

    'dilation 5x5' : np.array([[1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1]]),
        
    'soften 3x3' : (1 / 16.0) * np.array([[1, 2, 1],
                                          [2, 4, 2],
                                          [1, 2, 1]]),
    
    'soften 5x5' : (1 / 256.0) * np.array([[1,  4,  6,  4,  1],
                                           [4, 16, 24, 16,  4],
                                           [6, 24, 36, 24,  6],
                                           [4, 16, 24, 16,  4],
                                           [1,  4,  6,  4,  1]]),
        
    'scharr horizontal edge 3x3' : np.array([[-3,  0, 3],
                                         [-10, 0, 10],
                                         [-3,  0, 3]]),

    'scharr horizontal edge 5x5' : np.array([[-1, -3,  0, 3,  1],
                                             [-3, -10, 0, 10, 3],
                                             [-4, -12, 0, 12, 4],
                                             [-3, -10, 0, 10, 3],
                                             [-1, -3,  0, 3,  1]]),
         
    'scharr vertical edge 3x3' : np.array([[-3, -10,  -3],
                                           [ 0,  0,    0],
                                           [ 3,  10,   3]]),

    'scharr vertical edge 5x5' : np.array([[-1, -3,  -4,   -3, -1],
                                           [-3, -10, -12, -10, -3],
                                           [ 0,  0,   0,   0,   0],
                                           [ 3,  10,  12,  10,  3],
                                           [ 1,  3,   4,   3,   1]]),

    'motion blur 3x3' : (1 / 9.0) * np.array([[1,  0,   0],
                                              [0,  1,   0],
                                              [0,  0,   1]]),
    
    'motion blur 5x5' : (1 / 9.0) * np.array([[1,  0, 0, 0,  1],
                                              [0,  1, 0, 1,  0],
                                              [0,  0, 1, 0,  0],
                                              [0,  1, 0, 1,  0],
                                              [1,  0, 0, 0,  1]]),

    'robert horizontal edge 3x3': np.array([[0, 0,  0],
                                            [0, 1,  0],
                                            [0, 0, -1]]),

    'robert horizontal edge 5x5': np.array([[0, 0, 0,  0, 0],
                                            [0, 0, 0,  0, 0],
                                            [0, 0, 1,  0, 0],
                                            [0, 0, 0, -1, 0],
                                            [0, 0, 0,  0, 0]]),

    'robert vertical edge 3x3': np.array([[0,  0, 0],
                                          [0,  0, 1],
                                          [0, -1, 0]]),

    'robert vertical edge 5x5': np.array([[0, 0,  0, 0, 0],
                                          [0, 0,  0, 0, 0],
                                          [0, 0,  0, 1, 0],
                                          [0, 0, -1, 0, 0],
                                          [0, 0,  0, 0, 0]]),

    'ridge detection1 3x3' : np.array([[0, -1, 0], # or edge detection1 3x3
                                       [-1, 4,-1],
                                       [0, -1, 0]]),

    'ridge detection1 5x5' : np.array([[ 0,  0, -1,  0,  0], # or edge detection1 5x5
                                       [ 0, -1,  4, -1,  0],
                                       [-1,  4,  20, 4, -1],
                                       [ 0, -1,  4, -1,  0],
                                       [ 0,  0, -1,  0,  0]]),

    'ridge detection2 3x3' : np.array([[-1, -1, -1], # or edge detection2 3x3
                                       [-1,  8, -1],
                                       [-1, -1, -1]]),

    'ridge detection2 5x5' : np.array([[-1, -1, -1, -1, -1], # or edge detection2 5x5
                                       [-1, -1, -1, -1, -1],
                                       [-1, -1,  8, -1, -1],
                                       [-1, -1, -1, -1, -1],
                                       [-1, -1, -1, -1, -1]])
    }


class ApplyKernels():
    """
        To reach a wide range of effects and enhancements in digital images, You can use this class in various image processing tasks.


        Attributes:
        - random_seed: int, default=42
            To get the same results each time you run the function, You should initialize it with your desired value.

        Methods:
        - apply_filter_on_gray_img: Apply some kernel(s) on a grayscale image.
        - apply_filter_on_color_img: Apply some kernel(s) on a RGB color-scale image.
        - imshow: Show the input image.
    """
    def __init__(self, random_seed:int=42):
        self.random_seed = random_seed



    def __get_filtered_image(self, X, kernel_name, kernel_size, kernel_value):
        """
        Filter the input image using the provided kernel.
        """
        if kernel_name.lower() != 'custom' and type(kernel_size) == tuple:
            filtered_image = cv2.filter2D(src=X, ddepth=-1, kernel=kernels[kernel_name])

            return filtered_image

        elif kernel_name.lower() == 'custom' and kernel_size == None:
            filtered_image = cv2.filter2D(src=X, ddepth=-1, kernel=kernel_value)

            return filtered_image



    def __implementation(self, X, kernel_name, kernel_size, kernel_value):
        """
        It's used for applying filter(s) on an image using the private method __get_filtered_image. So, It returns a filtered image or a dictionary of filtered images
        """
        if kernel_value == None:
            valid_kernel_size = [3, 5]

            if len(X.shape) == 2 or len(X.shape) == 3: 
                if (kernel_size[0] in valid_kernel_size) and (kernel_size[1] in valid_kernel_size) and (kernel_size[0] == kernel_size[1]):
                    if type(kernel_name) == list:
                        if len(kernel_name) != 0:
                            try:
                                k_name = kernel_name[0].lower()

                            except:
                                raise AttributeError(f'{type(kernel_name[0])} object has no attribute `lower`')
                            
                            else:
                                del k_name
                                k_names = [
                                    k_name.lower() + ' ' + f'{kernel_size[0]}x{kernel_size[1]}'
                                    for k_name in kernel_name
                                    if (k_name.lower() + ' ' + f'{kernel_size[0]}x{kernel_size[1]}') in kernels.keys()
                                    ] 

                                if len(k_names) == 0:
                                    raise ValueError('There is no valid kernel name. For more info, please read the Docstring :)')
                                
                                elif len(k_names) == 1:
                                    kernel_name = k_names[0]

                                else:
                                    filtered_images = {}
                                    for k_name in k_names:
                                        filtered_image = self.__get_filtered_image(X, k_name, kernel_size, kernel_value)
                                        filtered_images[k_name] = filtered_image

                                        return filtered_images # a dictionary of various filters of an image       
                        else:
                            raise ValueError('There is no item in the kernel_name parameter')
                        

                    if type(kernel_name) == str and kernel_name.lower() == 'all':
                        filtered_images = {}
                        for k_name in kernels.keys():
                            filtered_image = self.__get_filtered_image(X, k_name, kernel_size, kernel_value)
                            filtered_images[k_name] = filtered_image

                        return filtered_images # a dictionary of various filters of an image 
                    
                    elif type(kernel_name) == str and ( (kernel_name.lower() + ' ' + f'{kernel_size[0]}x{kernel_size[1]}') in kernels.keys() ):
                        filtered_image = self.__get_filtered_image(X, (kernel_name + ' ' + f'{kernel_size[0]}x{kernel_size[1]}'), kernel_size, kernel_value)

                        return filtered_image # a 2-D Array

                    else:
                        if type(kernel_name) == tuple:
                            raise TypeError(f'kernel_name must be a string or list, Not {type(kernel_name)}. For more info, Please read the Docstring :)')
                        
                        else:
                            raise KeyError(f'There is no kernel named {kernel_name}. For more info, please read the Docstring :)')
                        
                else:
                    raise ValueError(f'Expected {(kernel_size[0], kernel_size[0])}, not {kernel_size}!')

            else:
                return 0
        


    def apply_filter_on_gray_img(self, X, kernel_name, kernel_size:tuple=None, kernel_value=None):
        """
        Apply some kernel(s) on a grayscale image.
            
            
        Parameters:
        ------------
        - X: `numpy.ndarray` (a 2-D array) e.g. (499, 635)
            The grayscale image on which the filter(s) will be applied.
            
        - kernel_name: str | list
            Kernel name can be a string such as 'all', 'custom', or one of kernels listed below or a list of kernels listed below.

            The list of valid kernels:
            'guassian blur' : The blur kernel applies a smoothing effect, averaging the pixel values in the neighborhood.
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
            'dilation' : The dilation kernel expands regions of bright pixels in an image.
            'soften' : The soften kernel is used to reduce image noise and create a smoother appearance while preserving overall image details.
            'scharr horizontal edge': The scharr horizontal edge kernel is used for edge detection and gradient estimation along the horizontal direction. It provides more weight to the central pixel and its immediate neighbors.	
            'scharr vertical edge': The scharr vertical edge kernel is used for edge detection and gradient estimation along the vertical direction. It provides more weight to the central pixel and its immediate neighbors.
            'motion blur' : The motion blur kernel is used to simulate the effect of motion in an image. It achieves this by applying a linear blur in a specific direction. The kernel consists of non-zero values along a line in the direction of motion, with zeros elsewhere.
            'robert horizontal edge' : A simple and efficient edge detection operator to detect horizontal edges. The kernels consist of positive and negative values that highlight the change in intensity along the respective directions.
            'robert vertical edge' : A simple and efficient edge detection operator to detect vertical edges. The kernels consist of positive and negative values that highlight the change in intensity along the respective directions.
            'ridge detection1' : In this kernel, the weights are increased towards the center to emphasize the ridge-like structures in the image.
            'ridge detection2' : This kernel is designed to enhance ridge-like structures in the image. The central value is higher (8) to emphasize the ridge, while the surrounding values are negative (-1) to suppress the surrounding areas.

        - kernel_size: tuple | NoneType, default=None
            The kernel which will be applied to the input image must be a square-like shape such as (3, 3) or (5, 5).

        - kernel_value: `numpy.ndarray` (a 2-D array) | NoneType, default=None
            When the kernel value is None, It means the builtin kernels are supposed to be used as filters because there is a value for kernel size and kernel name.   
            When the kernel value is a numpy array, It means the provided kernel are supposed to be used as a filter because the kernel name is custom. 

        Returns:
        ------------
        `numpy.ndarray`(a 2-D Array), Dict
        """
        if type(kernel_value) == np.ndarray:
            if len(kernel_value.shape) == 2:
                if kernel_size != None:
                    kernel_size = None
                
                if kernel_name.lower() != 'custom':
                    kernel_name = 'custom'

                if kernel_size == None and kernel_name.lower() == 'custom':
                    if type(X) == np.ndarray:
                        if len(X.shape) == 2:
                            np.random.seed(self.random_seed)  
                            filtered_image = self.__get_filtered_image(X, kernel_name, kernel_size, kernel_value)

                            return filtered_image
                        
                        else:
                            raise ValueError(f'Expected a 2-D array, but got an array with the shape of {X.shape}')
                                    
                    else:
                        raise TypeError(f'X must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')

            else:
                raise ValueError(f'Expected a 2-D array as a kernel, but got an array with the shape of {kernel_value.shape}')
        
        elif kernel_value == None:
            if type(kernel_size) == tuple:
                if type(X) == np.ndarray:
                    if len(X.shape) == 2:
                        if type(kernel_name) == str or type(kernel_name) == list:
                            np.random.seed(self.random_seed)  
                            filtered_image = self.__implementation(X, kernel_name, kernel_size, kernel_value)

                            if type(filtered_image) == int and filtered_image == 0:
                                raise ValueError(f'Expected 2 axes, but got {len(X.shape)}!')
                            
                            else:
                                return filtered_image # a grayscale image or a dictioary of grayscale images
                            
                        else:
                            raise TypeError(f'kernel_name must be a string or list, Not {type(kernel_name)}. For more info, Please read the Docstring :)')
                        
                    else:
                        raise ValueError(f'Expected a 2-D array, but got an array with the shape of {X.shape}')
                    
                else:
                    raise TypeError(f'X must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')

            else:
                raise TypeError(f'Expected a tuple object, Not {type(kernel_size)}. For more info, Please read the Docstring :)')
            
        else:
            raise ValueError(f'kernel_value must be a matrix or None, Not {type(kernel_value)}. For more info, Please read the Docstring :)')
        


    def apply_filter_on_color_img(self, X, kernel_name, kernel_size:tuple=None, kernel_value=None, with_resize:bool=False, dsize='auto'):
        """
        Apply some kernel(s) on a RGB color-scale image.
            
            
        Parameters:
        ------------
        - X: array-like e.g. (499, 635, 3)
            The color-scale image on which the filter(s) will be applied.
            
        - kernel_name: str | list
            Kernel name can be a string such as 'all', 'custom', or one of kernels listed below or a list of kernels listed below.

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
                'dilation' : The dilation kernel expands regions of bright pixels in an image.
                'soften' : The soften kernel is used to reduce image noise and create a smoother appearance while preserving overall image details.
                'scharr horizontal edge': The scharr horizontal edge kernel is used for edge detection and gradient estimation along the horizontal direction. It provides more weight to the central pixel and its immediate neighbors.	
                'scharr vertical edge': The scharr vertical edge kernel is used for edge detection and gradient estimation along the vertical direction. It provides more weight to the central pixel and its immediate neighbors.
                'motion blur' : The motion blur kernel is used to simulate the effect of motion in an image. It achieves this by applying a linear blur in a specific direction. The kernel consists of non-zero values along a line in the direction of motion, with zeros elsewhere.
                'robert horizontal edge' : A simple and efficient edge detection operator to detect horizontal edges. The kernels consist of positive and negative values that highlight the change in intensity along the respective directions.
                'robert vertical edge' : A simple and efficient edge detection operator to detect vertical edges. The kernels consist of positive and negative values that highlight the change in intensity along the respective directions.
                'ridge detection1' : In this kernel, the weights are increased towards the center to emphasize the ridge-like structures in the image.
                'ridge detection2' : This kernel is designed to enhance ridge-like structures in the image. The central value is higher (8) to emphasize the ridge, while the surrounding values are negative (-1) to suppress the surrounding areas.

        - kernel_size: tuple | NoneType, default=None
            The kernel which will be applied to the input image must be a square-like shape such as (3, 3) or (5, 5).

        - kernel_value: `numpy.ndarray` (a 2-D array) | NoneType, default=None
            When the kernel value is None, It means the builtin kernels are supposed to be used as filters because there is a value for kernel size and kernel name.   
            When the kernel value is a numpy array, It means the provided kernel are supposed to be used as a filter because the kernel name is custom. 

        - with_resize: bool, default=False
            To improve the speed of rendering matrices, You can use this parameter.
            1. If the number of pixels in the height and width of an input image is equal to or bigger than 400 pixels, and If you have assigned True to the with_resize parameter, The number of pixels will change to `[img.height * 0.5]` in height, and `[img.width * 0.5]` in width.
            2. If the number of pixels in the height and width of an input image is smaller than 400 pixels, and If you have assigned True to the with_resize parameter, The number of pixels won't change because the number of pixels in height and width are smaller than the expected value, 400 pixels.

        - disze: str or tuple, default='auto'
            You can use this parameter to resize the dimensions of an image when you are assigning the True value to the with_resize parameter.
            If the value of dsize is 'auto', It will use the formula `[img.width * 0.5]` to change the number of pixels in width, and use `[img.height * 0.5]` to change the number of pixels in height.
            If the type of dsize is tuple and contains 2 integer values, It will use disze[0] and disze[1] to change the number of pixels in width and height.

        Returns:
        ------------
        `numpy.ndarray`(a 2-D Array), Dict
        """
        def main_operations(func, X, kernel_name, kernel_size:tuple=None, kernel_value=None, with_resize:bool=False, dsize='auto'):
            try: 
                axis2_val = X.shape[2]
                
            except:
                raise IndexError(f'Expected 3 axes but got {len(X.shape)}!')
            
            else:
                del axis2_val
                if X.shape[2] == 3 and len(X.shape) == 3:
                    np.random.seed(self.random_seed) 

                    if with_resize == True:
                        if type(dsize) == str:
                            if dsize == 'auto':
                                if X.shape[0] >= 400 and X.shape[1] >= 400:
                                    X = cv2.resize(X, (round(X.shape[0] * .5), round(X.shape[1] * .5)))
                                    filtered_image = func(X, kernel_name, kernel_size, kernel_value)

                                    return filtered_image # a color-scale image or a dictioary of color-scale images
                                
                                else:
                                    with_resize = False

                            else:
                                raise ValueError('desize can only be `auto` in the string format')
                            
                        elif type(dsize) == tuple:
                            if len(dsize) == 2:
                                if type(dsize[0]) == int and type(dsize[1]) == int:
                                    X = cv2.resize(X, dsize)
                                    filtered_image = func(X, kernel_name, kernel_size, kernel_value)
                                    return filtered_image # a color-scale image or a dictioary of color-scale images
                                else:
                                    raise TypeError('dsize must be a tuple and contains 2 integer values')
                            else:
                                raise ValueError(f'Expected a tuple with length 2, Not {len(dsize)}!')
                            
                        else:
                            raise TypeError(f'dsize must be a string or tuple, Not {type(dsize)} with value {dsize}. For mor info, Please read the Docstring :)')
                        
                    if with_resize == False:
                        filtered_image = func(X, kernel_name, kernel_size, kernel_value)
                        return filtered_image # a color-scale image or a dictioary of color-scale images
                    
                else:
                    raise ValueError(f'Expected 3 axes and 3 channels but got {len(X.shape)} axes and {X.shape[2]} channels!')

        if type(kernel_value) == np.ndarray:
            if len(kernel_value.shape) == 2:
                if kernel_size != None:
                    kernel_size = None
                
                if kernel_name.lower() != 'custom':
                    kernel_name = 'custom'

                if kernel_size == None and kernel_name.lower() == 'custom':
                    if type(X) == np.ndarray:
                        filtered_image = main_operations(self.__get_filtered_image, X, kernel_name, kernel_size, kernel_value, with_resize, dsize)

                        return filtered_image
                                    
                    else:
                        raise TypeError(f'X must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')

            else:
                raise ValueError(f'Expected a 2-D array as a kernel, but got an array with the shape of {kernel_value.shape}')        

        elif kernel_value == None:
            if type(kernel_size) == tuple:
                if type(X) == np.ndarray:
                    if type(kernel_name) == str or type(kernel_name) == list:
                        filtered_image = main_operations(self.__implementation, X, kernel_name, kernel_size, kernel_value, with_resize, dsize)

                        return filtered_image
                            
                    else:
                        raise TypeError(f'kernel_name must be a string or list, Not {type(kernel_name)}. For more info, Please read the Docstring :)')
                    
                else:
                    raise TypeError(f'X must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')

            else:
                raise TypeError(f'Expected a tuple object, Not {type(kernel_size)}. For more info, Please read the Docstring :)')

        else:
            raise ValueError(f'kernel_value must be a matrix or None, Not {type(kernel_value)}. For more info, Please read the Docstring :)')



    def imshow(self, image, figsize:tuple=(5, 5), cmap=None):
        """
        Show the input image.

             
        Parameters:
        ------------
        - image: array-like or PIL image
            The image data
        
        - figsize: tuple, default=(5, 5)
            The figure size
        
        - cmap: str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            It maps scalar data to colors.
        """
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
        plt.tight_layout()

