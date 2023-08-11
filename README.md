# Pyimkernel
<p>
With this package, You can apply various image kernels such as Blur, Sobel, Scharr and so forth (The list of image kernels is mentioned below) on a gray scale or color image, and show images using the class ApplyKernels in this package to reach a wide range of effects and enhancements in digital images.
</p>

## Installation
<pre>
pip install pyimkernel
</pre>

## Usage
```python
from pyimkernel import ApplyKernels
import mnist

# Load data
X_train, X_test, y_train, y_test = mnist.train_images(), mnist.test_images(), mnist.train_labels(), mnist.test_labels()

# Create an instance
imkernel = ApplyKernels(random_seed=0)

# Show image 9 
imkernel.imshow(X_train[19], cmap=plt.cm.gray)

# Apply blur kernel on a gray scale image 9
filtered_image = imkernel.apply_filter_on_gray_img(X_train[19], kernel_name='blur')

# Show the filtered image
imkernel.imshow(image=filtered_image, cmap=plt.cm.gray)
```
## Output
<b>Before</b> Applying the blur kernel on a gray scale image 9

<br /><br/>
<img src="https://i.postimg.cc/Bn8nRVyY/image9.png">
<br /><br/>


<b>After</b> Applying the blur kernel on a gray scale image 9
<br /><br/>
<img src="https://i.postimg.cc/qBWrzxvS/filtered-image9.png">
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

