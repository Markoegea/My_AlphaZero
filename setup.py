#!/usr/bin/env python
from setuptools import setup, find_packages

DISTNAME = 'PMAZ'
DESCRIPTION = 'Project_MyAlphaZero'
MAINTAINER = 'Marco A. Egea Moreno'
URL = 'https://github.com/Markoegea/my_alphazero'

classifiers = ['Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.10.11']

if __name__ == "__main__":
    setup(name=DISTNAME,
          version='0.1',
          packages=find_packages(),
          maintainer=MAINTAINER,
          description=DESCRIPTION,
          url=URL,
          classifiers=classifiers,
          install_requires=[
                'certifi',
                'charset-normalizer',
                'colorama',
                'contourpy',
                'cycler',
                'filelock',
                'fonttools',
                'idna',
                'Jinja2',
                'kiwisolver',
                'MarkupSafe',
                'matplotlib',
                'mpmath',
                'networkx',
                'numpy',
                'packaging',
                'Pillow',
                'pyparsing',
                'python-dateutil',
                'requests',
                'six',
                'sympy',
                'torch',
                'torchaudio',
                'torchvision',
                'tqdm',
                'typing_extensions',
                'urllib3',
          ],
          license='MIT')