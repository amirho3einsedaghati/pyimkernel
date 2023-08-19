# Pyimkernel
[![License: MIT](https://img.shields.io/github/license/amirho3einsedaghati/pyimkernel?color=yellow)](https://github.com/amirho3einsedaghati/pyimkernel/blob/main/LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/amirho3einsedaghati/pyimkernel?color=red)](https://github.com/amirho3einsedaghati/pyimkernel/)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/amirho3einsedaghati/pyimkernel?color=yellow)](https://github.com/amirho3einsedaghati/pyimkernel/pulls)
[![GitHub issues](https://img.shields.io/github/issues-raw/amirho3einsedaghati/pyimkernel?color=red)](https://github.com/amirho3einsedaghati/pyimkernel/issues)

<p>
With this package, You can apply various image kernels such as Blur, Sobel, Scharr and so forth (The comprehensive list of image kernels is mentioned below) on a grayscale or color-scale image, and then show them. These effects and enhancements in digital images can be achieved using the "ApplyKernels" class, allowing for a wide range of transformations.
</p>

## Image kernels
<b> Image kernels are listed below:</b>

- blur
- bottom sobel
- emboss
- identity
- left sobel
- outline
- right sobel
- sharpen
- top sobel
- horizontal edge
- vertical edge
- box blur
- laplacian
- prewitt horizontal edge
- prewitt vertical edge
- high-pass filter
- unsharp masking
- dilate
- soften
- scharr horizontal edge
- scharr vertical edge
- motion blur

## Installation
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

# Create an instance
imkernel = ApplyKernels(random_seed=0)

# Grayscale
# Show image 9 
imkernel.imshow(X_train[19], cmap=plt.cm.gray)

# Apply blur kernel on a grayscale image 9
filtered_image = imkernel.apply_filter_on_gray_img(X_train[19], kernel_name='blur')

# Show the filtered image 9
imkernel.imshow(image=filtered_image, cmap='gray')

# the Color-Scale image
# Read the flower image
image1 = cv2.imread(os.path.join('Images', '1.jpg'))

# Show the flower image
# Convert a color-scale image to a grayscale one and then visualize it
imkernel.imshow(cv2.cvtColor(image1, cv2.COLOR_BGRA2GRAY), cmap='gray', figsize=(6, 6))

# Show the filtered flower image
blurred_image = imkernel.apply_filter_on_color_img(image1, kernel_name='box blur', with_resize=True, dsize=(100, 100))
imkernel.imshow(image=blurred_image)

sharpened_image = imkernel.apply_filter_on_color_img(image1, kernel_name='sharpen', with_resize=True)
imkernel.imshow(image=sharpened_image, figsize=(6, 6))

imkernel.imshow(image=imkernel.apply_filter_on_color_img(sharpened_image, kernel_name='soften'),
                figsize=(6, 6))
```
## The Grayscale Output
<p><b>Before</b> Applying the Blur Kernel on a grayscale image 9</p>
<img src="https://i.postimg.cc/m23gBQW3/image9.png">
<br /><br/>

<p><b>After</b> Applying the Blur Kernel on a grayscale image 9</p>
<img src="https://i.postimg.cc/BvPWQ01W/filtered-image9.png">

## The Color Scale Output
<p><b>Before</b> Applying the Box Blur Kernel on a color-scale flower image</p>
<img src="https://i.postimg.cc/QMGm1GyZ/flower.png">
<br /><br/>

<p><b>After</b> Applying the Box Blur Kernel on a color-scale flower image and assigning True to the with_resize parameter and (100, 100) to the dsize parameter</p>
<img src="https://i.postimg.cc/fWByFZ8V/filtered-flower3.png">
<br /><br/>

<p><b>After</b> Applying the Sharpen Kernel on a color-scale flower image and assigning True to the with_resize parameter and 'auto' to the dsize parameter</p>
<img src="https://i.postimg.cc/3NCRJQTc/filtered-flower1.png">
<br /><br/>

<p><b>After</b> Applying the Soften Kernel on a color-scale image, which was filtered using the Sharpen Blur Kernel before, and assigning True to the with_resize parameter</p>
<img src="https://i.postimg.cc/KvjSYyx3/filtered-flower2.png">

## Package Status
All Tests Passed.

## Maintainer Contact
<a href="https://linktr.ee/amirhoseinsedaghati">Link</a>
