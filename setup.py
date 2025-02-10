#!/usr/bin/env python
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='total-perspective-vortex',
    version='0.1',
    description='brain computer interface based on electroencephalographic data',
    long_description=readme,
    author='Armandase',
    url='https://github.com/Armandase/total-perspective-vortex/',
    license=license,
    install_requires=['mne', 'scikit-learn', 'matplotlib', 'numpy']
)
