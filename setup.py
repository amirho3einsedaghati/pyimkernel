from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    f.close()

setup(
    name='pyimkernel',
    version='1.2.2',
    packages=find_packages(exclude=['tests*', 'examples*']),
    license='MIT',
    description='Applying some image kernel(s) on a grayscale or RGB color-scale image',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy', 'matplotlib', 'opencv-python'],
    url='https://github.com/amirho3einsedaghati/pyimkernel',
    author='Amir Hosein Sedaghati',
    author_email='amirhosseinsedaghati42@gmail.com',
    #maintainer_contact='https://linktr.ee/amirhoseinsedaghati',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering'       
    ],
    keywords=['image kernels/filters', 'image preprocessing', 'Applying image kernels/filters on a grayscale or color scale image']
)
