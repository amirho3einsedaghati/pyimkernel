from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    f.close()
    
setup(
    name='pyimkernel',
    version='0.5.0',
    packages=find_packages(exclude=['tests*', 'examples*']),
    license='MIT',
    description='Applying some image kernel(s) on an image',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy', 'matplotlib', 'cv2'],
    url='https://github.com/amirho3einsedaghati/pyimkernel',
    author='Amir Hosein Sedaghati',
    author_email='amirhosseinsedaghati42@gmail.com',
    author_contact='https://linktr.ee/amirhoseinsedaghati',
    classifiers=['Programming Language :: Python :: 3'],
    keywords=['image kernels/filters', 'image preprocessing', 'Applying image kernels/filters on a grayscale or color scale image']
)
