# LibLLIE: Low-Light Image Enhancement Library

## Introduction

`LibLLIE` is an open-source library **for Low-Light Image Enhancement** built on PyTorch.

​	This Python library provides a collection of traditional machine learning algorithms for low-light image enhancement. The library supports multiple color spaces and offers various methods for image enhancement. It is designed to be modular and easy to integrate into your existing projects.

​	The integration code for deep learning models is currently being developed and will be released soon !

​	Feel free to use this library in your research!

⭐ Star us on GitHub — your support means a lot!

## Table of Contents

------

- [Installation](#installation)

- Implementations

  - HE series(HE、CLAHE、RCLAHE)

  - Math series(gamma correction、log transformation)

  - Dark Channel Prior(DCP)

  - Multiple Color Spaces

    ······

    (More algorithms will be released soon !)

## Installation

------

There are two ways to install `LibLLIE`.

```python
conda create -n LlieLib python=3.8 -y
pip install -r requirements.txt
```

**Method 1: Install from Source Code**

```python
git clone https://github.com/glory-wan/LLIE-Lib.git

pip install -e .  # Editable mode, suitable for those who wish to modify the source code.
pip install .     # Standard mode, suitable for installing the package without the need for further modification of the source code.
```

**Method 2: Install Using a Compressed Package**

With this method, you don't need to download the source code. Simply download a compressed package and execute the following command in your local Python environment.

The compressed package can be found at [`LLIE-Lib/dist`](https://github.com/glory-wan/LLIE-Lib/tree/main/dist).

```
pip install LibLlie-1.0.tar.gz
```

## Quick Start

------

### Command-Line Interface

To quickly start processing images using the command-line interface:

```
python example/commandTA.py --img path/to/img --method he --cs hsv --name he_hsv --display True
```

Supported algorithms and color spaces can be found in the `LibLlie/troditionAlgorithm/config.py` file.