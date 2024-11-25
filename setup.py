from setuptools import setup, find_packages

setup(
    name='gensurv',
    version='0.1.1',
    description='A simple Python module to share functions',
    author='Leetkue',
    author_email='your.email@example.com',
    url='https://github.com/leon-etienne/gensurv',  # GitHub URL
    packages=find_packages(),
    install_requires=[
        'scikit-image',
        'ffmpegcv',
        'opencv-python',
        'transformers',
        'diffusers',
        'pillow',
        'ultralytics'
    ],  # Add any dependencies here
)
