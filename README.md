# Pyimkernel
[![License: MIT](https://img.shields.io/github/license/amirho3einsedaghati/pyimkernel?color=yellow)](https://github.com/amirho3einsedaghati/pyimkernel/blob/main/LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/amirho3einsedaghati/pyimkernel?color=red)](https://github.com/amirho3einsedaghati/pyimkernel/)
![GitHub forks](https://img.shields.io/github/forks/amirho3einsedaghati/pyimkernel?color=yellow)
![GitHub User's stars](https://img.shields.io/github/stars/amirho3einsedaghati/pyimkernel?color=red)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/amirho3einsedaghati/pyimkernel?color=yellow)](https://github.com/amirho3einsedaghati/pyimkernel/pulls)
[![GitHub issues](https://img.shields.io/github/issues-raw/amirho3einsedaghati/pyimkernel?color=red)](https://github.com/amirho3einsedaghati/pyimkernel/issues)

<p>
With this package, You can apply various image kernels such as Blur, Sobel, Scharr, and so forth (The comprehensive list of image kernels is mentioned below) on a grayscale or RGB|BGR color-scale image, and then show them, or manipulate the pixels of PIL images. These effects and enhancements in digital images can be achieved using the <b>ApplyKernels</b> and <b>ManipulatePixels</b> class, allowing for a wide range of transformations.

<b> This package helps you:</b>
- to understand what happens when applying kernels to images.
- to address the issue of limited data by increasing image quantities.
- to develop end-to-end applications.


If you are interested in learning more about the package, I encourage you to check out the [main.py](https://github.com/amirho3einsedaghati/pyimkernel/blob/main/pyimkernel/main.py) file.
</p>

## About Dependencies
To develop this package, I used the following libraries:
- <b>numpy</b> : for working with arrays and Kernels.
- <b>matplotlib</b> : for visualizing images.
- <b>opencv-python</b> : for working with array-like images and applying kernels to images.
- <b>pillow</b> : for working with PIL images.
- <b>pilgram2</b> : for working with Equivalent and Blending techniques.
   
## Image kernels
You can use built-in image kernels which have the shape of (3, 3) or (5, 5) or use your desired kernel value. To see how you can apply kernels to images, You can check out the [usage section](https://github.com/amirho3einsedaghati/pyimkernel?tab=readme-ov-file#usage) or the file [Image Kernels.ipynb](https://github.com/amirho3einsedaghati/pyimkernel/blob/main/examples/Image%20Kernels.ipynb).

<b> The supported Built-in Image kernels are listed below:</b>
- <b>guassian blur or blur</b> : The blur kernel applies a smoothing effect, averaging the pixel values in the neighborhood.
- <b>bottom sobel</b> : The bottom sobel kernel emphasizes edges in the bottom directions.
- <b>emboss</b> : The emboss kernel creates a 3D embossed effect in the image.
- <b>identity</b> : The identity kernel does not modify the image and leaves it unchanged.
- <b>left sobel</b> : The left sobel kernel emphasizes edges in the left directions.
- <b>outline</b> : The outline kernel detects edges and boundaries by emphasizing the differences in intensity between neighboring pixels.
- <b>right sobel</b> : The right sobel kernel emphasizes edges in the right directions.
- <b>sharpen</b> : The sharpen kernel enhances edges and details in an image. 
- <b>top sobel</b> : The top sobel kernel emphasizes edges in the top directions.
- <b>horizontal edge</b> : The horizontal edge kernel highlights horizontal edges.
- <b>vertical edge</b> : The vertical edge kernel highlights vertical edges.
- <b>box blur</b> : The box blur kernel is similar to the blur kernel. It applies a simple averaging operation to create a blur effect, but with equal weights     for all neighboring pixels.
- <b>laplacian</b> : The Laplacian kernel is used for edge detection and image sharpening.
- <b>prewitt horizontal edge</b> : The prewitt horizontal edge kernel is similar to the bottom sobel kernel, emphasizing edges in the horizontal directions.
- <b>prewitt vertical edge</b> : The prewitt vertical edge kernel is similar to the right sobel kernel, emphasizing edges in the horizontal directions.
- <b>high-pass filter</b> : The high-pass filter kernel enhances the details and edges in an image while reducing the low-frequency components.
- <b>unsharp masking</b> : The unsharp masking kernel is used for image sharpening. It enhances the edges and details in an image by subtracting a blurred version of the image from the original.
- <b>dilation</b> : The Dilation kernels expand the bright regions or foreground in an image, which can make color images appear closer to white.
- <b>soften</b> : The soften kernel is used to reduce image noise and create a smoother appearance while preserving overall image details.
- <b>scharr horizontal edge</b> : The scharr horizontal edge kernel is used for edge detection and gradient estimation along the horizontal direction. It provides more weight to the central pixel and its immediate neighbors.
- <b>scharr vertical edge</b> : The scharr vertical edge kernel is used for edge detection and gradient estimation along the vertical direction. It provides more weight to the central pixel and its immediate neighbors.
- <b>motion blur</b> : The motion blur kernel is used to simulate the effect of motion in an image. It achieves this by applying a linear blur in a specific direction. The kernel consists of non-zero values along a line in the direction of motion, with zeros elsewhere.
- <b>robert horizontal edge</b> : A simple and efficient edge detection operator to detect horizontal edges. The kernels consist of positive and negative values that highlight the change in intensity along the respective directions.
- <b>robert vertical edge</b> : A simple and efficient edge detection operator to detect vertical edges. The kernels consist of positive and negative values that highlight the change in intensity along the respective directions.
- <b>ridge detection1 or edge detection1</b>: In this kernel, the weights are increased towards the center to emphasize the ridge-like structures in the image.
- <b>ridge detection2 or edge detection2</b>: This kernel is designed to enhance ridge-like structures in the image. The central value is higher (8) to emphasize the ridge, while the surrounding values are negative (-1) to suppress the surrounding areas.

## Equivalent and Blending Techniques
Using equivalent and blending techniques, you can manipulate the pixels of PIL images. To see their mathematic formulas, read the Docstring provided for each in the [main.py](https://github.com/amirho3einsedaghati/pyimkernel/blob/main/pyimkernel/main.py) file. 

<b> The supported equivalent techniques are listed below:</b>
- <b>brightness</b> : Adjusts the brightness.
- <b>contrast</b> : Adjusts the contrast.
- <b>grayscale</b> : Converts image to grayscale.
- <b>saturate</b> : Saturates image.
- <b>sepia</b> : Converts image to sepia.
- <b>hue_rotate</b> : Applies hue rotation.
  
<b> The supported blending techniques are listed below:</b>
- <b>color</b> : Creates a color with the hue and saturation of the source color and the luminosity of the backdrop color.
- <b>color_burn</b> : Darkens the backdrop color to reflect the source color.
- <b>color_dodge</b> : Brightens the backdrop color to reflect the source color.
- <b>darken</b> : Selects the darker of the backdrop and source colors.
- <b>difference</b> : Subtracts the darker of the two constituent colors from the lighter color.
- <b>exclusion</b> : Produces an effect like Difference but lower in contrast.
- <b>hard_light</b> : Multiplies or screens the colors, depending on the source color value.
- <b>hue</b> : Creates a color with the hue of the source color and the saturation and luminosity of the backdrop color.
- <b>lighten</b> : Selects the lighter of the backdrop and source colors.
- <b>multiply</b> : The source color is multiplied by the destination color and replaces the destination.
- <b>normal</b> : The normal Technique simply selects the source color.
- <b>overlay</b> : Multiplies or screens the colors, depending on the backdrop color value.
- <b>screen</b> : Multiplies the complements of the backdrop and source color values, then complements the result.
- <b>soft_light</b> : Darkens or lightens the colors, depending on the source color value.
  
## Pyimkernel Installation
<pre>
pip install pyimkernel
</pre>

## Usage
```python
from pyimkernel import ApplyKernels
import mnist # pip install mnist
import cv2
import os

# Load data
X_train, X_test, y_train, y_test = mnist.train_images(), mnist.test_images(), mnist.train_labels(), mnist.test_labels()

######################################### Apply Kernels to images #########################################

# Create an instance from ApplyKernels
imkernel = ApplyKernels(random_seed=0)

# Grayscale
# Show image 9 
imkernel.imshow(
  X_train[19],
  cmap=plt.cm.gray
)

# Apply the 3x3 blur kernel to a grayscale image 9
filtered_image = imkernel.apply_filter_on_gray_img(
  X_train[19],
  kernel_name='blur',
  kernel_size=(3, 3)
)

# Show the filtered image 9
imkernel.imshow(
  image=filtered_image,
  cmap='gray'
)

# 5x5 Prewitt Horizontal Edge Kernel
kernel_value = np.array([[-1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1],
                         [0,   0,  0,  0,  0],
                         [1,   1,  1,  1,  1],
                         [1,   1,  1,  1,  1]])

# First apply the 5x5 Prewitt kernel to image 9 and then show the filtered image
imkernel.imshow(
    image=imkernel.apply_filter_on_gray_img(
        X_train[19],
        kernel_name='custom',
        kernel_value=kernel_value
    ),
    cmap=plt.cm.gray
)

# the Color-Scale image
# Read the flower image
image1 = cv2.imread(
    os.path.join('Images', '1.jpg')
)

# Show the flower image
imkernel.imshow(
    cv2.cvtColor(image1, cv2.COLOR_BGR2RGB),
    figsize=(5, 5)
)

# Show the filtered flower image
# 5x5 box blur
kernel_value = (1 / 25.0) * np.array([[1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1]])
# Apply the 5x5 box blur kernel to image1
blurred_image = imkernel.apply_filter_on_color_img(
    image1,
    kernel_name='custom',
    kernel_value=kernel_value,
    with_resize=True,
    dsize=(100, 100)
)
# Show the blurred image
imkernel.imshow(
    image=cv2.cvtColor(
        blurred_image,
        cv2.COLOR_BGR2RGB
    ))

# Apply the 3x3 sharpen kernel to image1
sharpened_image = imkernel.apply_filter_on_color_img(
    image1,
    kernel_name='sharpen',
    kernel_size=(3, 3),
    with_resize=True
)
# Show the sharpened image
imkernel.imshow(
    image=cv2.cvtColor(
        sharpened_image,
        cv2.COLOR_BGR2RGB
    ))

# First Apply the 5x5 soften kernel to a sharpened image and then show the filtered image
imkernel.imshow(
    image=cv2.cvtColor(
        imkernel.apply_filter_on_color_img(
            sharpened_image, 
            kernel_name='soften',
            kernel_size=(5, 5)
        ),
    cv2.COLOR_BGR2RGB
    ))

# First Apply the 3x3 unsharp masking kernel on a sharpened image and then show the filtered image
imkernel.imshow(
    image=cv2.cvtColor(
        imkernel.apply_filter_on_color_img(
            sharpened_image,
            kernel_name='unsharp masking',
            kernel_size=(3, 3)
        ),
    cv2.COLOR_BGR2RGB
    ))

######################################### Manipulate the pixels of PIL images #########################################

# Create an instance from ManipulatePixels
manipulation = ManipulatePixels()

# read the Sid image
image2 = cv2.imread(
    os.path.join('Images', 'ice-age.jpeg')
)
# show the Sid image
imkernel.imshow(
    cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
)

# manipulate and show the Sid image using the equivalent technique `hue_rotate`
im = manipulation.hue_rotate(
    Image.fromarray(
        cv2.cvtColor(
            image2, cv2.COLOR_BGR2RGB
        )),
    degree=30
)
manipulation.imshow(im)

# manipulate and show the Sid image using the blending technique `color`
im = manipulation.color(
    Image.fromarray(image2),
    Image.fromarray(
        cv2.resize(
            cv2.cvtColor(
                image1, cv2.COLOR_BGR2RGB
            ),
        image2.shape[:2])
    ))
manipulation.imshow(im)
```
## After using the ApplyKernels class
<pre><b>The Grayscale Output</b></pre>
<p><b>Before</b> Applying Kernels on a grayscale image 9</p>
<img src="https://i.postimg.cc/HL0LV9xG/image9.png">
<br /><br/>

<p><b>After</b> Applying the 3x3 Blur Kernel on a grayscale image 9</p>
<img src="https://i.postimg.cc/HkVZY66x/filtered-image9.png">

<p><b>After</b> Applying the 5x5 Prewitt Horizontal Edge Kernel on a grayscale image 9</p>
<img src="https://i.postimg.cc/Nj2CRKrR/filtered-nine2.png">

<pre><b>The Color-scale Output</b></pre>
<p><b>Before</b> Applying Kernels on a color-scale flower image</p>
<img src="https://i.postimg.cc/tJgqY01w/1.jpg">
<br /><br/>

<p><b>After</b> Applying the `5x5 Box Blur Kernel` on a color-scale flower image and assigning True to the with_resize parameter and (100, 100) to the dsize parameter</p>
<img src="https://i.postimg.cc/rp6njqS3/filtered-flower1.png">
<br /><br/>

<p><b>After</b> Applying the `3x3 sharpen kernel` on a color-scale flower image and assigning True to the with_resize parameter and 'auto' to the dsize parameter</p>
<img src="https://i.postimg.cc/26G0CvPN/filtered-flower2.png">
<br /><br/>

<p><b>After</b> Applying the `5x5 soften kernel` on a color-scale image, which was filtered using the `Sharpen Kernel` before, and assigning False to the with_resize parameter</p>
<img src="https://i.postimg.cc/cH0DqTkc/filtered-flower3.png">
<br /><br/>

<p><b>After</b> Applying the `3x3 unsharp masking` kernel on a color-scale image, which was filtered using the `Sharpen Kernel` before, and assigning False to the with_resize parameter</p>
<img src="https://i.postimg.cc/PJq2JJjp/filtered-flower4.png">

## After using the ManipulatePixels class

<p><b>Before</b> Using Equivalent and Blending Techniques</p>
<img src="https://i.postimg.cc/L6fF2cY2/orgin-sid.png">

<p><b>After</b> Using the Equivalent technique `hue_rotate` for manipulating the pixels of Sid image</p>
<img src="https://i.postimg.cc/43WrWZMb/filtered-sid1.png">

<p><b>After</b> Using the Blending technique `color` for manipulating the pixels of Sid image</p>
<img src="https://i.postimg.cc/jq7F7zsh/filtered-sid2.png">

## Package Status
All Tests Passed Successfully.

<p><b>After</b> applying the built-in image kernels to the Sid image using the ApplyKernels class</p>
<img src="https://i.postimg.cc/TwtpRzhg/Sid.png">

<p><b>After</b> applying the Equivalent techniques to the Sid image using the ManipulatePixels class</p>
<img src="https://i.postimg.cc/XqC6zmSH/equivalent-techniques.png">

<p><b>After</b> applying the Blending techniques to the Sid image using the ManipulatePixels class</p>
<img src="https://i.postimg.cc/L5wLHqW4/blending-techniques.png">

## About References
To develop the package, I used the [Setosa](https://setosa.io/ev/image-kernels/), [Wikipedia](https://en.wikipedia.org/wiki/Kernel_(image_processing)), and [ChatGPT](https://chat.openai.com/) websites to gather kernel values and utilized other websites to ensure that the information I collected was accurate and properly attributed. 

## Appreciation
I want to thank [@sugizo](https://github.com/sugizo) for providing valuable and insightful ideas that helped me improve the package.

## Maintainer Contact
<a href="https://linktr.ee/amirhoseinsedaghati">https://linktr.ee/amirhoseinsedaghati</a>
