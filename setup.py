from setuptools import setup, find_packages

setup(
    name='LibLlie',
    version='1.0',
    packages=find_packages(),
    author='Guangrong Wan',
    author_email='glory947446@gmail.com',
    description='A Python library for low-light image enhancement, '
                'integrating various algorithms and deep learning models '
                'to improve image visibility and quality under low-light conditions.',
    keywords='low-light enhancement, deep learning,'
             'computer vision, low-light conditions,'
             'image processing, image enhancement',
    install_requires=[
            'opencv-python==4.9.0.80',
            'numpy==1.24.1',
            'matplotlib==3.7.4',
            'requests==2.28.1',
            'pillow==10.2.0',
            'setuptools==68.2.2',
            'scipy==1.10.1',
    ]
)
