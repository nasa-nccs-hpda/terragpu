#!/usr/bin/env python

from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='xrasterlib',
    version='0.0.1',
    description='Methods for remote sensing satellite imagery processing',
    author='Jordan A. Caraballo-Vega',
    author_email='jordan.a.caraballo-vega@nasa.gov',
    zip_safe=False,
    url='https://github.com/jordancaraballo/xrasterlib.git',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
