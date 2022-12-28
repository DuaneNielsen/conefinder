#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='conefinder',
      version='1.0',
      description='I finds out cones',
      author='Duane Nielsen',
      author_email='duane.nielsen.rocks@gmail.com',
      packages=find_packages('.'),
      package_data={
            "conefinder": ["resources/*"]
      },
      entry_points={'console_scripts': ['conefinder=conefinder.scripts.main:main']},
      install_requires=['cython', 'numpy', 'opencv-python', 'pyopengl'],
     )
