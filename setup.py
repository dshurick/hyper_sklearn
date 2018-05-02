#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'scikit-learn', 'sklearn', 'hyperopt', 'numpy', 'xgboost']

setup_requirements = []

test_requirements = []

setup(
    author="Devon Shurick",
    author_email='dshurick@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Bayesian Optimization for Hyperparameter Tuning",
    entry_points={
        'console_scripts': [
            'hyper_sklearn=hyper_sklearn.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='hyper_sklearn',
    name='hyper_sklearn',
    packages=find_packages(include=['hyper_sklearn']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/dshurick/hyper_sklearn',
    version='0.1.0',
    zip_safe=False,
)
