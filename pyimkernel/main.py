import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pilgram2.css.blending import *
import pilgram2.css.brightness as brightness
import pilgram2.css.contrast as contrast
import pilgram2.css.grayscale as grayscale
import pilgram2.css.saturate as saturate
import pilgram2.css.sepia as sepia
import pilgram2.css.hue_rotate as hue_rotate
from pyimkernel.builtin_kernels import kernels



class ApplyKernels():
    """To reach a wide range of effects and enhancements in digital images,
    You can use this class in various image processing tasks.

    Attributes:
    ------------
        - random_seed: int, default=42
            To get the same results each time you run the function,
            You should initialize it with your desired value.

    Methods:
    ------------
        - apply_filter_on_gray_img:
            Apply some kernel(s) on a grayscale image.

        - apply_filter_on_color_img:
            Apply some kernel(s) on a RGB color-scale image.

        - imshow:
            Show the input image.
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
        It's used for applying filter(s) on an image using the private method __get_filtered_image.
        So, It returns a filtered image or a dictionary of filtered images
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
                                    raise ValueError(f'There is no valid `kernel_name` becaue got {k_names[0]}. For more info, please read the Docstring :)')
                                
                                elif len(k_names) == 1:
                                    kernel_name = k_names[0]

                                else:
                                    filtered_images = {}
                                    for k_name in k_names:
                                        filtered_image = self.__get_filtered_image(X, k_name, kernel_size, kernel_value)
                                        filtered_images[k_name] = filtered_image

                                        return filtered_images # a dictionary of various filters of an image       
                        else:
                            raise ValueError('There is no item in the `kernel_name` parameter')
                        

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
                            raise TypeError(f'`kernel_name` must be a string or list, Not {type(kernel_name)}. For more info, Please read the Docstring :)')
                        
                        else:
                            raise KeyError(f'There is no kernel named {kernel_name}. For more info, please read the Docstring :)')
                        
                else:
                    raise ValueError(f'Expected {(kernel_size[0], kernel_size[0])}, not {kernel_size}!')

            else:
                return 0
        


    def apply_filter_on_gray_img(self, X, kernel_name, kernel_size:tuple=None, kernel_value=None):
        """Apply some kernel(s) on a grayscale image.
            
        Parameters:
        ------------
        - X: `numpy.ndarray` (a 2-D array) e.g. (499, 635)
            The grayscale image on which the filter(s) will be applied.
            
        - kernel_name: str | list
            kernel_name can be a string such as 'all', 'custom', or one
            of the kernels listed below or a list of the kernels listed below.

            When kernel_name is set to 'custom', you only need to determine
            the value of the kernel_value argument and don't need to pass a
            value to the kernel_size argument because in this case, the
            value for kernel_size is None.

            When kernel_name is set to 'all', one of the built-in kernels or
            a list of the kernels listed below , you only need to determine
            the value of the kernel_size argument and don't need to pass a
            value to the kernel_value argument because in this case, the
            value for kernel_value is None.

            The list of built-in kernels:

            'guassian blur' :
                The blur kernel applies a smoothing effect, averaging the pixel
                values in the neighborhood.

            'bottom sobel' :
                The bottom sobel kernel emphasizes edges in the bottom directions.

            'emboss' :
                The emboss kernel creates a 3D embossed effect in the image.

            'identity' :
                The identity kernel does not modify the image and leaves it unchanged.

            'left sobel' :
                The left sobel kernel emphasizes edges in the left directions.

            'outline' :
                The outline kernel detects edges and boundaries by emphasizing
                the differences in intensity between neighboring pixels.

            'right sobel' :
                The right sobel kernel emphasizes edges in the right directions.

            'sharpen' :
                The sharpen kernel enhances edges and details in an image.

            'top sobel' :
                The top sobel kernel emphasizes edges in the top directions.

            'horizontal edge' :
                The horizontal edge kernel highlights horizontal edges.

            'vertical edge' :
                The vertical edge kernel highlights vertical edges.

            'box blur' :
                The box blur kernel is similar to the blur kernel. It applies
                a simple averaging operation to create a blur effect, but with
                equal weights for all neighboring pixels.

            'laplacian' :
                The Laplacian kernel is used for edge detection and image sharpening.

            'prewitt horizontal edge' :
                The prewitt horizontal edge kernel is similar to the bottom
                sobel kernel, emphasizing edges in the horizontal directions.

            'prewitt vertical edge' :
                The prewitt vertical edge kernel is similar to the right sobel kernel,
                emphasizing edges in the horizontal directions.

            'high-pass filter' :
                The high-pass filter kernel enhances the details and edges in an image
                while reducing the low-frequency components.

            'unsharp masking' :
                The unsharp masking kernel is used for image sharpening. It enhances
                the edges and details in an image by subtracting a blurred version of
                the image from the original.

            'dilation' :
                The Dilation kernels expand the bright regions or foreground in an
                image, which can make color images appear closer to white.

            'soften' :
                The soften kernel is used to reduce image noise and create a smoother
                appearance while preserving overall image details.

            'scharr horizontal edge':
                The scharr horizontal edge kernel is used for edge detection and
                gradient estimation along the horizontal direction. It provides
                more weight to the central pixel and its immediate neighbors.

            'scharr vertical edge':
                The scharr vertical edge kernel is used for edge detection and gradient
                estimation along the vertical direction. It provides more weight to the
                central pixel and its immediate neighbors.

            'motion blur' :
                The motion blur kernel is used to simulate the effect of motion in an
                image. It achieves this by applying a linear blur in a specific
                direction.The kernel consists of non-zero values along a line in
                the direction of motion, with zeros elsewhere.

            'robert horizontal edge' :
                A simple and efficient edge detection operator to detect horizontal
                edges. The kernels consist of positive and negative values that
                highlight the change in intensity along the respective directions.

            'robert vertical edge' :
                A simple and efficient edge detection operator to detect vertical
                edges. The kernels consist of positive and negative values that
                highlight the change in intensity along the respective directions.

            'ridge detection1' :
                In this kernel, the weights are increased towards the center to
                emphasize the ridge-like structures in the image.

            'ridge detection2' :
                This kernel is designed to enhance ridge-like structures in the image.
                The central value is higher (8) to emphasize the ridge, while the
                surrounding values are negative (-1) to suppress the surrounding areas.

        - kernel_size: tuple | NoneType, default=None
            The kernel which will be applied to the input image must be a square-like
            shape such as (3, 3) or (5, 5).

        - kernel_value: `numpy.ndarray` (a 2-D array) | NoneType, default=None
            When the kernel value is None, It means the builtin kernels are supposed
            to be used as filters because there is a value for kernel size and kernel
            name.

            When the kernel value is a numpy array, It means the provided kernel are
            supposed to be used as a filter because the kernel name is custom.

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
                        raise TypeError(f'`X` must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')

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
                            raise TypeError(f'`kernel_name` must be a string or list, Not {type(kernel_name)}. For more info, Please read the Docstring :)')
                        
                    else:
                        raise ValueError(f'Expected a 2-D array, but got an array with the shape of {X.shape}')
                    
                else:
                    raise TypeError(f'`X` must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')

            else:
                raise TypeError(f'Expected a tuple object, Not {type(kernel_size)}. For more info, Please read the Docstring :)')
            
        else:
            raise ValueError(f'`kernel_value` must be a matrix or None, Not {type(kernel_value)}. For more info, Please read the Docstring :)')
        


    def apply_filter_on_color_img(self, X, kernel_name, kernel_size:tuple=None, kernel_value=None, with_resize:bool=False, dsize='auto'):
        """Apply some kernel(s) on a RGB or BGR color-scale image. 
            
        Parameters:
        ------------
        - X: array-like e.g. (499, 635, 3)
            The color-scale image on which the filter(s) will be applied.
            
         - kernel_name: str | list
            kernel_name can be a string such as 'all', 'custom', or one
            of the kernels listed below or a list of the kernels listed below.

            When kernel_name is set to 'custom', you only need to determine
            the value of the kernel_value argument and don't need to pass a
            value to the kernel_size argument because in this case, the
            value for kernel_size is None.

            When kernel_name is set to 'all', one of the built-in kernels or
            a list of the kernels listed below , you only need to determine
            the value of the kernel_size argument and don't need to pass a
            value to the kernel_value argument because in this case, the
            value for kernel_value is None.

            The list of built-in kernels:

            'guassian blur' :
                The blur kernel applies a smoothing effect, averaging the pixel
                values in the neighborhood.

            'bottom sobel' :
                The bottom sobel kernel emphasizes edges in the bottom directions.

            'emboss' :
                The emboss kernel creates a 3D embossed effect in the image.

            'identity' :
                The identity kernel does not modify the image and leaves it unchanged.

            'left sobel' :
                The left sobel kernel emphasizes edges in the left directions.

            'outline' :
                The outline kernel detects edges and boundaries by emphasizing
                the differences in intensity between neighboring pixels.

            'right sobel' :
                The right sobel kernel emphasizes edges in the right directions.

            'sharpen' :
                The sharpen kernel enhances edges and details in an image.

            'top sobel' :
                The top sobel kernel emphasizes edges in the top directions.

            'horizontal edge' :
                The horizontal edge kernel highlights horizontal edges.

            'vertical edge' :
                The vertical edge kernel highlights vertical edges.

            'box blur' :
                The box blur kernel is similar to the blur kernel. It applies
                a simple averaging operation to create a blur effect, but with
                equal weights for all neighboring pixels.

            'laplacian' :
                The Laplacian kernel is used for edge detection and image sharpening.

            'prewitt horizontal edge' :
                The prewitt horizontal edge kernel is similar to the bottom
                sobel kernel, emphasizing edges in the horizontal directions.

            'prewitt vertical edge' :
                The prewitt vertical edge kernel is similar to the right sobel kernel,
                emphasizing edges in the horizontal directions.

            'high-pass filter' :
                The high-pass filter kernel enhances the details and edges in an image
                while reducing the low-frequency components.

            'unsharp masking' :
                The unsharp masking kernel is used for image sharpening. It enhances
                the edges and details in an image by subtracting a blurred version of
                the image from the original.

            'dilation' :
                The Dilation kernels expand the bright regions or foreground in an
                image, which can make color images appear closer to white.

            'soften' :
                The soften kernel is used to reduce image noise and create a smoother
                appearance while preserving overall image details.

            'scharr horizontal edge':
                The scharr horizontal edge kernel is used for edge detection and
                gradient estimation along the horizontal direction. It provides
                more weight to the central pixel and its immediate neighbors.

            'scharr vertical edge':
                The scharr vertical edge kernel is used for edge detection and gradient
                estimation along the vertical direction. It provides more weight to the
                central pixel and its immediate neighbors.

            'motion blur' :
                The motion blur kernel is used to simulate the effect of motion in an
                image. It achieves this by applying a linear blur in a specific
                direction.The kernel consists of non-zero values along a line in
                the direction of motion, with zeros elsewhere.

            'robert horizontal edge' :
                A simple and efficient edge detection operator to detect horizontal
                edges. The kernels consist of positive and negative values that
                highlight the change in intensity along the respective directions.

            'robert vertical edge' :
                A simple and efficient edge detection operator to detect vertical
                edges. The kernels consist of positive and negative values that
                highlight the change in intensity along the respective directions.

            'ridge detection1' :
                In this kernel, the weights are increased towards the center to
                emphasize the ridge-like structures in the image.

            'ridge detection2' :
                This kernel is designed to enhance ridge-like structures in the image.
                The central value is higher (8) to emphasize the ridge, while the
                surrounding values are negative (-1) to suppress the surrounding areas.

        - kernel_size: tuple | NoneType, default=None
            The kernel which will be applied to the input image must be a square-like
            shape such as (3, 3) or (5, 5).

        - kernel_value: `numpy.ndarray` (a 2-D array) | NoneType, default=None
            When the kernel value is None, It means the builtin kernels are supposed
            to be used as filters because there is a value for kernel size and kernel
            name.

            When the kernel value is a numpy array, It means the provided kernel are
            supposed to be used as a filter because the kernel name is custom.

        - with_resize: bool, default=False
            To improve the speed of rendering matrices, You can use this parameter.

            1. If the number of pixels in the height and width of an input image is
               equal to or bigger than 400 pixels, and If you have assigned True to
               the with_resize parameter, The number of pixels will change to
              `[img.height * 0.5]` in height, and `[img.width * 0.5]` in width.

            2. If the number of pixels in the height and width of an input image
               is smaller than 400 pixels, and If you have assigned True to the
               with_resize parameter, The number of pixels won't change because
               the number of pixels in height and width are smaller than the
               expected value, 400 pixels.

        - disze: str or tuple, default='auto'
            You can use this parameter to resize the dimensions of an image when
            you are assigning the True value to the with_resize parameter.

            If the value of dsize is 'auto', It will use the formula
            `[img.width * 0.5]` to change the number of pixels in width,
            and use `[img.height * 0.5]` to change the number of pixels in height.

            If the type of dsize is tuple and contains 2 integer values,
            It will use disze[0] and disze[1] to change the number of pixels
            in width and height.

        Returns:
        ------------
        `numpy.ndarray`(a 2-D Array), Dict
        """
        def main_operations(func, X, kernel_name, kernel_size: tuple = None, kernel_value=None, with_resize: bool = False, dsize='auto'):
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

                                    return filtered_image  # a color-scale image or a dictioary of color-scale images

                                else:
                                    with_resize = False

                            else:
                                raise ValueError('`desize` can only be `auto` in the string format')

                        elif type(dsize) == tuple:
                            if len(dsize) == 2:
                                if type(dsize[0]) == int and type(dsize[1]) == int:
                                    X = cv2.resize(X, dsize)
                                    filtered_image = func(X, kernel_name, kernel_size, kernel_value)
                                    return filtered_image  # a color-scale image or a dictioary of color-scale images
                                else:
                                    raise TypeError('`dsize` must be a tuple and contains 2 integer values')
                            else:
                                raise ValueError(f'Expected a tuple with length 2, Not {len(dsize)}!')

                        else:
                            raise TypeError(
                                f'`dsize` must be a string or tuple, Not {type(dsize)} with value {dsize}. For mor info, Please read the Docstring :)')

                    if with_resize == False:
                        filtered_image = func(X, kernel_name, kernel_size, kernel_value)
                        return filtered_image  # a color-scale image or a dictioary of color-scale images

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
                        raise TypeError(f'`X` must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')

            else:
                raise ValueError(f'Expected a 2-D array as a kernel, but got an array with the shape of {kernel_value.shape}')        

        elif kernel_value == None:
            if type(kernel_size) == tuple:
                if type(X) == np.ndarray:
                    if type(kernel_name) == str or type(kernel_name) == list:
                        filtered_image = main_operations(self.__implementation, X, kernel_name, kernel_size, kernel_value, with_resize, dsize)

                        return filtered_image
                            
                    else:
                        raise TypeError(f'`kernel_name` must be a string or list, Not {type(kernel_name)}. For more info, Please read the Docstring :)')
                    
                else:
                    raise TypeError(f'`X` must be array-like, Not {type(X)}. For more info, Please read the Docstring :)')

            else:
                raise TypeError(f'Expected a tuple object, Not {type(kernel_size)}. For more info, Please read the Docstring :)')

        else:
            raise ValueError(f'`kernel_value` must be a matrix or None, Not {type(kernel_value)}. For more info, Please read the Docstring :)')



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



class  ManipulatePixels(ApplyKernels):
    """using Equivalent and Blending techniques, you can manipulate the pixels of PIL images.

    Methods:
    ------------
        - Equivalent Techniques:
            - brightness : Adjusts the brightness.
            - contrast : Adjusts the contrast.
            - grayscale : Converts image to grayscale.
            - saturate : Saturates image.
            - sepia : Converts image to sepia.
            - hue_rotate : Applies hue rotation.

        - Blending Techniques:
            - color :
                Creates a color with the hue and saturation of the source
                color and the luminosity of the backdrop color.

            - color_burn :
                Darkens the backdrop color to reflect the source color.

            - color_dodge :
                Brightens the backdrop color to reflect the source color.

            - darken :
                Selects the darker of the backdrop and source colors.

            - difference :
                Subtracts the darker of the two constituent colors from the
                lighter color.

            - exclusion :
                Produces an effect like Difference but lower in contrast.

            - hard_light :
                Multiplies or screens the colors, depending on the source
                color value.

            - hue :
                Creates a color with the hue of the source color and the
                saturation and luminosity of the backdrop color.

            - lighten :
                Selects the lighter of the backdrop and source colors.

            - multiply :
                The source color is multiplied by the destination color and
                replaces the destination.

            - normal :
                The normal Technique simply selects the source color.

            - overlay :
                Multiplies or screens the colors, depending on the backdrop
                color value.

            - screen :
                Multiplies the complements of the backdrop and source color
                values, then complements the result.

            - soft_light :
                Darkens or lightens the colors, depending on the source color
                value.

        - imshow: Show the input image.
    """
    def __init__(self):
        pass

    
    def __equivalent_techniques(self, func, image, amount=None, degree=None):
        """Using Equivalent techniques, we can manipulate the pixels of a PIL image.
        """
        if isinstance(image, Image.Image):
            try:
                if degree == None:
                    filtered_image = func(image, amount)

                elif amount == None:
                    filtered_image = func(image, degree)

            except AssertionError:
                raise ValueError(f"`amount` must be equal or bigger than 0, but got {amount}.")
            
            else:
                return filtered_image
            
        else:
            raise TypeError(f'Expected a PIL image, Not {type(image)}. For more info, Please read the Docstring :)')
        

    def brightness(self, image, amount=1):
        """Adjusts the brightness.

        A brightness operation is equivalent to the following matrix operation:

            | R' |     | c  0  0 |   | R |
            | G' |  =  | 0  c  0 | * | G |
            | B' |     | 0  0  c |   | B |

        See the W3C document:
        https://www.w3.org/TR/filter-effects-1/#brightnessEquivalent

        Arguments:
        ------------
            - image: PIL Image
                An input image.

            - amount: int | float, default=1
                The filter amount (percentage).
                
        Returns:
        ------------
            a PIL Image

        Raises:
        ------------
            ValueError: if `amount` is less than 0.
            TypeError: if `image` is not a PIL image.
        """
        return self.__equivalent_techniques(brightness, image, amount)


    def contrast(self, image, amount=1):
        """Adjusts the contrast.

        A contrast operation is equivalent to the following matrix operation:

            | R' |     | c  0  0  0  -0.5c+0.5 |   | R |
            | G' |     | 0  c  0  0  -0.5c+0.5 |   | G |
            | B' |  =  | 0  0  c  0  -0.5c+0.5 | * | B |
            | A' |     | 0  0  0  1          0 |   | A |
            | 1  |     | 0  0  0  0          1 |   | 1 |

        See the W3C document:
        https://www.w3.org/TR/filter-effects-1/#contrastEquivalent.

        Arguments:
        ------------
            - image: PIL Image
                An input image.

            - amount: int | float, default=1
                The filter amount (percentage).

        Returns:
        ------------
            a PIL Image

        Raises:
        ------------
            ValueError: if `amount` is less than 0.
            TypeError: if `image` is not a PIL image.
        """
        return self.__equivalent_techniques(contrast, image, amount)


    def grayscale(self, image, amount=1):
        """Converts image to grayscale.

        A grayscale operation is equivalent to the following matrix operation:

        | R' |     |0.2126+0.7874g  0.7152-0.7152g  0.0722-0.0722g 0  0 |   | R |
        | G' |     |0.2126-0.2126g  0.7152+0.2848g  0.0722-0.0722g 0  0 |   | G |
        | B' |  =  |0.2126-0.2126g  0.7152-0.7152g  0.0722+0.9278g 0  0 | * | B |
        | A' |     |            0               0               0  1  0 |   | A |
        | 1  |     |            0               0               0  0  1 |   | 1 |

        See the W3C document:
        https://www.w3.org/TR/filter-effects-1/#grayscaleEquivalent

        Arguments:
        ------------
            - image: PIL Image
                An input image.

            - amount: int | float, default=1
                The filter amount (percentage).

        Returns:
        ------------
            a PIL Image

        Raises:
        ------------
            ValueError: if `amount` is less than 0.
            TypeError: if `image` is not a PIL image.
        """
        return self.__equivalent_techniques(grayscale, image, amount)


    def hue_rotate(self, image, degree=0):
        """Applies hue rotation.

        A hue rotate operation is equivalent to the following matrix operation:

            | R' |     | a00  a01  a02  0  0 |   | R |
            | G' |     | a10  a11  a12  0  0 |   | G |
            | B' |  =  | a20  a21  a22  0  0 | * | B |
            | A' |     | 0    0    0    1  0 |   | A |
            | 1  |     | 0    0    0    0  1 |   | 1 |

        where

            | a00 a01 a02 |    [+0.213 +0.715 +0.072]
            | a10 a11 a12 | =  [+0.213 +0.715 +0.072] +
            | a20 a21 a22 |    [+0.213 +0.715 +0.072]
                                    [+0.787 -0.715 -0.072]
            cos(hueRotate value) *  [-0.213 +0.285 -0.072] +
                                    [-0.213 -0.715 +0.928]
                                    [-0.213 -0.715+0.928]
            sin(hueRotate value) *  [+0.143 +0.140-0.283]
                                    [-0.787 +0.715+0.072]

        See the W3C document:
        https://www.w3.org/TR/SVG11/filters.html#feColorMatrixValuesAttribute

        Arguments:
        ------------
            - image: PIL Image
                An input image.

            - degree: int | float, default=1
                The hue rotate value (degrees).

        Returns:
        ------------
            a PIL Image.
        """
        return self.__equivalent_techniques(hue_rotate, image, degree=degree)


    def saturate(self, image, amount=1):
        """Saturates image.

        A saturate operation is equivalent to the following matrix operation:

            | R' |     |0.213+0.787s  0.715-0.715s  0.072-0.072s 0  0 |   | R |
            | G' |     |0.213-0.213s  0.715+0.285s  0.072-0.072s 0  0 |   | G |
            | B' |  =  |0.213-0.213s  0.715-0.715s  0.072+0.928s 0  0 | * | B |
            | A' |     |           0            0             0  1  0 |   | A |
            | 1  |     |           0            0             0  0  1 |   | 1 |

        See the W3C document:
        https://www.w3.org/TR/SVG11/filters.html#feColorMatrixValuesAttribute

        Arguments:
        ------------
            - image: PIL Image
                An input image.

            - amount: int | float, default=1
                The filter amount (percentage).

        Returns:
        ------------
            a PIL Image

        Raises:
        ------------
            ValueError: if `amount` is less than 0.
            TypeError: if `image` is not a PIL image.
        """
        return self.__equivalent_techniques(saturate, image, amount)
        

    def sepia(self, image, amount=1):
        """Converts image to sepia.

        A sepia operation is equivalent to the following matrix operation:

        | R' |     |0.393+0.607s  0.769-0.769s  0.189-0.189s 0  0 |   | R |
        | G' |     |0.349-0.349s  0.686+0.314s  0.168-0.168s 0  0 |   | G |
        | B' |  =  |0.272-0.272g  0.534-0.534g  0.131+0.869g 0  0 | * | B |
        | A' |     |          0             0             0  1  0 |   | A |
        | 1  |     |          0             0             0  0  1 |   | 1 |

        See the W3C document:
        https://www.w3.org/TR/filter-effects-1/#sepiaEquivalent

        Arguments:
        ------------
            - image: PIL Image
                An input image.

            - amount: int | float, default=1
                The filter amount (percentage).

        Returns:
        ------------
            a PIL Image

        Raises:
        ------------
            ValueError: if `amount` is less than 0.
            TypeError: if `image` is not a PIL image.
        """
        return self.__equivalent_techniques(sepia, image, amount)


    def __blending_techniques(self, func, backdrop_im, source_im):
        """Using Blending techniques, we can manipulate the pixels of PIL images.
        """
        if isinstance(backdrop_im, Image.Image) and isinstance(backdrop_im, Image.Image):
            try:
                filtered_image =  func(backdrop_im, source_im)
            
            except ValueError:
                raise ValueError(f"The shape of `source_im` is not the same as `backdrop_im`.\n\t    source_im : {np.asarray(source_im).shape} and backdrop_im : {np.asarray(backdrop_im).shape}")
            
            else:
                return filtered_image

        else:
            raise TypeError(f"Both `backdrop_im` and `source_im` must be PIL images.\n\t  `backdrop_im` : {type(backdrop_im)} and `source_im` : {type(source_im)}")


    def color(self, backdrop_im, source_im):
        """Creates a color with the hue and saturation of the source color
        and the luminosity of the backdrop color.

        The color formula is defined as:

            B(Cb, Cs) = SetLum(Cs, Lum(Cb))

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingcolor

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """ 
        return self.__blending_techniques(color, backdrop_im, source_im)


    def color_burn(self, backdrop_im, source_im):
        """Darkens the backdrop color to reflect the source color.

        The color burn formula is defined as:

            if(Cb == 1)
                B(Cb, Cs) = 1
            else if(Cs == 0)
                B(Cb, Cs) = 0
            else
                B(Cb, Cs) = 1 - min(1, (1 - Cb) / Cs)

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingcolorburn

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(color_burn, backdrop_im, source_im)
    

    def color_dodge(self, backdrop_im, source_im):
        """Brightens the backdrop color to reflect the source color.

        The color dodge formula is defined as:

            if(Cb == 0)
                B(Cb, Cs) = 0
            else if(Cs == 1)
                B(Cb, Cs) = 1
            else
                B(Cb, Cs) = min(1, Cb / (1 - Cs))

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingcolordodge

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(color_dodge, backdrop_im, source_im)


    def darken(self, backdrop_im, source_im):
        """Selects the darker of the backdrop and source colors.

        The darken formula is defined as:

            B(Cb, Cs) = min(Cb, Cs)

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingdarken

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(darken, backdrop_im, source_im)


    def difference(self, backdrop_im, source_im):
        """Subtracts the darker of the two constituent colors
        from the lighter color.

        The difference formula is defined as:

            B(Cb, Cs) = | Cb - Cs |

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingdifference

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(difference, backdrop_im, source_im)
        

    def exclusion(self, backdrop_im, source_im):
        """Produces an effect like Difference but lower in contrast.

        The exclusion formula is defined as:

            B(Cb, Cs) = Cb + Cs - 2 x Cb x Cs

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingexclusion

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(exclusion, backdrop_im, source_im)


    def hard_light(self, backdrop_im, source_im):
        """Multiplies or screens the colors, depending on the source color value.

        The hard light formula is defined as:

            if(Cs <= 0.5)
                B(Cb, Cs) = Multiply(Cb, 2 x Cs)
            else
                B(Cb, Cs) = Screen(Cb, 2 x Cs -1)

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendinghardlight

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(hard_light, backdrop_im, source_im)


    def hue(self, backdrop_im, source_im):
        """Creates a color with the hue of the source color
        and the saturation and luminosity of the backdrop color.

        The hue formula is defined as:

            B(Cb, Cs) = SetLum(SetSat(Cs, Sat(Cb)), Lum(Cb))

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendinghue

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(hue, backdrop_im, source_im)


    def lighten(self, backdrop_im, source_im):
        """Selects the lighter of the backdrop and source colors.

        The lighten formula is defined as:

            B(Cb, Cs) = max(Cb, Cs)

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendinglighten

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(lighten, backdrop_im, source_im)


    def multiply(self, backdrop_im, source_im):
        """The source color is multiplied by the destination color
        and replaces the destination.

        The mutiply formula is defined as:

            B(Cb, Cs) = Cb x Cs

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingmultiply

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(multiply, backdrop_im, source_im)


    def normal(self, backdrop_im, source_im):
        """The blending formula simply selects the source color.

        The normal formula is defined as:

            B(Cb, Cs) = Cs

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingnormal

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(normal, backdrop_im, source_im)


    def overlay(self, backdrop_im, source_im):
        """Multiplies or screens the colors, depending on the backdrop color value.

        The overlay formula is defined as:

            B(Cb, Cs) = HardLight(Cs, Cb)

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingoverlay

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(overlay, backdrop_im, source_im)


    def screen(self, backdrop_im, source_im):
        """Multiplies the complements of the backdrop and source color values,
        then complements the result.

        The screen formula is defined as:

            B(Cb, Cs) = 1 - [(1 - Cb) x (1 - Cs)]
                    = Cb + Cs - (Cb x Cs)

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingscreen

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(screen, backdrop_im, source_im)


    def soft_light(self, backdrop_im, source_im):
        """Darkens or lightens the colors, depending on the source color value.

        The soft light formula is defined as:

            if(Cs <= 0.5)
                B(Cb, Cs) = Cb - (1 - 2 x Cs) x Cb x (1 - Cb)
            else
                B(Cb, Cs) = Cb + (2 x Cs - 1) x (D(Cb) - Cb)

        where

            if(Cb <= 0.25)
                D(Cb) = ((16 * Cb - 12) x Cb + 4) x Cb
            else
                D(Cb) = sqrt(Cb)

        See the W3C document:
        https://www.w3.org/TR/compositing-1/#blendingsoftlight

        Arguments:
        ------------
            - backdrop_im: PIL Image
                A backdrop image (RGB or RGBA).

            - source_im:  PIL Image
                A source image (RGB or RGBA).

        Returns:
        ------------
            a PIL Image
        """
        return self.__blending_techniques(soft_light, backdrop_im, source_im)
    

    def imshow(self, image, figsize:tuple=(5, 5), cmap=None):
        super().imshow(image, figsize, cmap)  # Call the parent class function using the overriding method
