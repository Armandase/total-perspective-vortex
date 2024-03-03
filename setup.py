#!/usr/bin/env python
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='total-perspective-vortex',
    version='0.1',
    description='brain computer interface based on electroencephalographic data',
    long_description=readme,
    author='armandase',
    author_email='me@kennethreitz.com',
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)