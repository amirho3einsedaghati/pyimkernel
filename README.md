# Pyimkernel
[![License: MIT](https://img.shields.io/github/license/amirho3einsedaghati/pyimkernel?color=yellow)](https://github.com/amirho3einsedaghati/pyimkernel/blob/main/LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/amirho3einsedaghati/pyimkernel?color=red)](https://github.com/amirho3einsedaghati/pyimkernel/)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/amirho3einsedaghati/pyimkernel?color=yellow)](https://github.com/amirho3einsedaghati/pyimkernel/pulls)
[![GitHub issues](https://img.shields.io/github/issues-raw/amirho3einsedaghati/pyimkernel?color=red)](https://github.com/amirho3einsedaghati/pyimkernel/issues)

<p>
With this package, You can apply various image kernels such as Blur, Sobel, Scharr and so forth (The list of image kernels is mentioned below) on a grayscale or color-scale image, and then show them. All of these happens using the "ApplyKernels" class to reach a wide range of effects and enhancements in digital images.
</p>

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
imkernel.imshow(image1.reshape(image1.shape[0], -1), cmap='gray', figsize=(20, 10))

# Show the filtered flower image
blurred_image = imkernel.apply_filter_on_color_img(image1, kernel_name='motion blur', with_resize=True)
imkernel.imshow(image=imkernel.apply_filter_on_gray_img(blurred_image, kernel_name='sharpen'),
                figsize=(7, 6), cmap=plt.cm.gray)
```
## The Grayscale Output
<b>Before</b> Applying the blur kernel on a grayscale image 9
<img src="https://i.postimg.cc/Bn8nRVyY/image9.png">
<br /><br/>


<b>After</b> Applying the blur kernel on a grayscale image 9
<img src="https://i.postimg.cc/qBWrzxvS/filtered-image9.png">
<br /><br/>

## The Color Scale Output
<b>Before</b> Applying the laplacian kernel on a color-scale flower image
<img src="https://i.postimg.cc/B66PbRQk/flower.png">
<br /><br/>


<b>After</b> Applying the Sharpen Kernel on a filtered color-scale image using the Motion Blur Kernel and assigning True to the with_resize parameter
<img src="https://i.postimg.cc/x19P8V2B/filtered-flower.png">
<br /><br/>

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

<br /><br/>
<p>
All Tests Passed.
</p>
