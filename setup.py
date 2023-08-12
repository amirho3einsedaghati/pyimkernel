from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    f.close()
    
setup(
    name='pyimkernel',
    version='0.5.1',
    packages=find_packages(exclude=['tests*', 'examples*']),
    license='MIT',
    description='Applying some image kernel(s) on an image',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy', 'matplotlib', 'opencv-python'],
    url='https://github.com/amirho3einsedaghati/pyimkernel',
    author='Amir Hosein Sedaghati',
    author_email='amirhosseinsedaghati42@gmail.com',
    maintainer_contact='https://linktr.ee/amirhoseinsedaghati',
    classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Developers :: Education',
    'License :: OSI Approved :: MIT License',
    'Operating System :: MacOS :: NT :: POSIX :: UNIX',
    'Programming Language :: Python :: 3 :: Only Python :: 3.6 :: 3.7 :: 3.8 :: 3.9 :: 3.10 :: 3.11',
    'Topic :: Scientific/Engineering :: Image Preprocessing'
    ],
    keywords=['image kernels/filters', 'image preprocessing', 'Applying image kernels/filters on a grayscale or color scale image']
    Requires='Python >=3.6'
)
