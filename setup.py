from setuptools import setup, find_packages

from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

setup(
    name='bindpredict',
    version='0.0.1a.dev0',

    author="Nazanin Donyapour",
    author_email="nazanin@msu.edu",
    description="efscoring",
    license="MIT",
    url="https://github.com/ndonyapour/bindpredict",
    classifiers=[
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3'
    ],

  # package
    packages=find_packages(where='src'),

    package_dir={'' : 'src'},

    # if this is true then the package_data won't be included in the
    # dist. Use MANIFEST.in for this
    include_package_data=True,


    # SNIPPET: this is a general way to do this
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    entry_points = {},

    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'pytorch',
        'rdkit'
    ],
)
