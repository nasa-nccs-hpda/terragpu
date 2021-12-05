#!/usr/bin/env python

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='terragpu',
    version='21.11',
    description='Accelerating AI/ML in remote sensing satellite imagery',
    author='Jordan A. Caraballo-Vega',
    author_email='jordan.a.caraballo-vega@nasa.gov',
    zip_safe=False,
    url='https://github.com/nasa-nccs/terragpu.git',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'dl_pipeline = terragpu.ai.deep_learning.console.dl_pipeline:main',
            'rf_pipeline = terragpu.ai.machine_learning.console.rf_pipeline:main',
        ]
    },
)