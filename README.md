# Pyimkernel
[![License: MIT](https://img.shields.io/github/license/amirho3einsedaghati/pyimkernel?color=yellow)](https://github.com/amirho3einsedaghati/pyimkernel/blob/main/LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/amirho3einsedaghati/pyimkernel?color=red)](https://github.com/amirho3einsedaghati/pyimkernel/)
![GitHub forks](https://img.shields.io/github/forks/amirho3einsedaghati/pyimkernel?color=yellow)
![GitHub User's stars](https://img.shields.io/github/stars/amirho3einsedaghati/pyimkernel?color=red)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/amirho3einsedaghati/pyimkernel?color=yellow)](https://github.com/amirho3einsedaghati/pyimkernel/pulls)
[![GitHub issues](https://img.shields.io/github/issues-raw/amirho3einsedaghati/pyimkernel?color=red)](https://github.com/amirho3einsedaghati/pyimkernel/issues)

<p>
With this package, You can apply various image kernels such as Blur, Sobel, Scharr and so forth (The comprehensive list of image kernels is mentioned below) on a grayscale or RGB color-scale image, and then show them. These effects and enhancements in digital images can be achieved using the "ApplyKernels" class, allowing for a wide range of transformations.

<b> This package helps you:</b>
- to understand what happens when applying kernels to images
- address the issue of limited data by increasing image quantities.
</p>

## Image kernels
<b> The supported Image kernels are listed below:</b><br>
<br>
To see the definition of each following Kernels, you can read the Docstring provided in the main.py file.

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
imkernel.imshow(
  X_train[19],
  cmap=plt.cm.gray
)

# Apply blur kernel on a grayscale image 9
filtered_image = imkernel.apply_filter_on_gray_img(
  X_train[19],
  kernel_name='blur'
)

# Show the filtered image 9
imkernel.imshow(
  image=filtered_image,
  cmap='gray'
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
# Apply a box blur kernel on image1
blurred_image = imkernel.apply_filter_on_color_img(
    image1,
    kernel_name='box blur',
    with_resize=True,
    dsize=(100, 100)
)
# Show the blurred image
imkernel.imshow(
    image=cv2.cvtColor(
        blurred_image,
        cv2.COLOR_BGR2RGB
    ))

# Apply a sharpen kernel on image1
sharpened_image = imkernel.apply_filter_on_color_img(
    image1,
    kernel_name='sharpen',
    with_resize=True
)
# Show the sharpened image
imkernel.imshow(
    image=cv2.cvtColor(
        sharpened_image,
        cv2.COLOR_BGR2RGB
    ))

# First Apply a soften kernel on sharpened_image and then show the filtered image
imkernel.imshow(
    image=cv2.cvtColor(
        imkernel.apply_filter_on_color_img(
            sharpened_image, 
            kernel_name='soften'
        ),
    cv2.COLOR_BGR2RGB
    ))

# First Apply an unsharp masking kernel on sharpened_image and then show the filtered image
imkernel.imshow(
    image=cv2.cvtColor(
        imkernel.apply_filter_on_color_img(
            sharpened_image,
            kernel_name='unsharp masking'
        ),
    cv2.COLOR_BGR2RGB
    ))
```
## The Grayscale Output
<p><b>Before</b> Applying a Blur Kernel on a grayscale image 9</p>
<img src="https://i.postimg.cc/HL0LV9xG/image9.png">
<br /><br/>

<p><b>After</b> Applying a Blur Kernel on a grayscale image 9</p>
<img src="https://i.postimg.cc/HkVZY66x/filtered-image9.png">

## The Color Scale Output
<p><b>Before</b> Applying a Box Blur Kernel on a color-scale flower image</p>
<img src="https://i.postimg.cc/tJgqY01w/1.jpg">
<br /><br/>

<p><b>After</b> Applying a Box Blur Kernel on a color-scale flower image and assigning True to the with_resize parameter and (100, 100) to the dsize parameter</p>
<img src="https://i.postimg.cc/wBvYX5xR/filtered-flower1.png">
<br /><br/>

<p><b>After</b> Applying a Sharpen Kernel on a color-scale flower image and assigning True to the with_resize parameter and 'auto' to the dsize parameter</p>
<img src="https://i.postimg.cc/vZnVQ9w9/filtered-flower2.png">
<br /><br/>

<p><b>After</b> Applying a Soften Kernel on a color-scale image, which was filtered using the Sharpen Kernel before, and assigning False to the with_resize parameter</p>
<img src="https://i.postimg.cc/T354fkP9/filtered-flower3.png">
<br /><br/>

<p><b>After</b> Applying an Unsharp Masking Kernel on a color-scale image, which was filtered using the Sharpen Kernel before, and assigning False to the with_resize parameter</p>
<img src="https://i.postimg.cc/fWj3gZj7/filtered-flower4.png">

## Package Status
All Tests Passed Successfully.

## Maintainer Contact
<a href="https://linktr.ee/amirhoseinsedaghati">Links</a>
