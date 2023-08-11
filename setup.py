from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    f.close()
    
setup(
    name='pyimkernel',
    version='0.4.0',
    packages=find_packages(exclude=['tests*', 'examples*']),
    license='MIT',
    description='Apply image kernel(s) on a image',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy', 'matplotlib'],
    url='https://github.com/amirho3einsedaghati/pyimkernel',
    author='Amir Hosein Sedaghati',
    author_email='amirhosseinsedaghati42@gmail.com'
)
