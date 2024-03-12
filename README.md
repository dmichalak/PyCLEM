# pyclem

[![License](https://img.shields.io/pypi/l/pyclem.svg?color=green)](https://github.com/andreasmarnold/pyclem/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyclem.svg?color=green)](https://pypi.org/project/pyclem)
[![Python Version](https://img.shields.io/pypi/pyversions/pyclem.svg?color=green)](https://python.org)
[![CI](https://github.com/andreasmarnold/pyclem/actions/workflows/ci.yml/badge.svg)](https://github.com/andreasmarnold/pyclem/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/andreasmarnold/pyclem/branch/main/graph/badge.svg)](https://codecov.io/gh/andreasmarnold/pyclem)

Package containing tools and napari-widgets for the analysis of CLEM experiments.

# Installation
1. In git-bash:
    - clone the pyclem repository to a local hard-drive. To do so, open git bash in the desired location and run
    ```
    git clone https://git.lobos.nih.gov/taraskalab/pyclem.git
    ```
2. In Anaconda-prompt
    - Create a new virtual environment containing Python>=3.8
    ```
    conda create -y -n pyclem-env python=3.10
    ```
    - Activate the new environment
    ```
    conda activate pyclem-env
    ```
    - Navigate to the folder containing your clone of the PyCLEM repository.
    ```
    cd <your_folder_structure>/pyclem
    ```
    - Install the project with
    ```
    python -m pip install -e .
    ```


    - At this point in time, the Napari plugin "Affinder" is not published in the version we need. Therefore, install it directly from it's github repository.
    ```
    python -m pip install git+https://github.com/jni/affinder.git
    ```

