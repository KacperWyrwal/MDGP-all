#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

requirements = [
    "torch", 
    "gpytorch",
    "pymanopt",
    "botorch",
    "geoopt",
    "lightning",
    "plotly",
    "coclust @ git+https://github.com/KacperWyrwal/cclust_package-MDGP.git@master", 
    "geometrickernels @ git+https://github.com/KacperWyrwal/GeometricKernels-torch-vectorized.git@devel",
]

setup(
    name="mdgp",
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
)
