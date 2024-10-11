# pyclem

[![License](https://img.shields.io/pypi/l/pyclem.svg?color=green)](https://github.com/andreasmarnold/pyclem/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyclem.svg?color=green)](https://pypi.org/project/pyclem)
[![Python Version](https://img.shields.io/pypi/pyversions/pyclem.svg?color=green)](https://python.org)
[![CI](https://github.com/andreasmarnold/pyclem/actions/workflows/ci.yml/badge.svg)](https://github.com/andreasmarnold/pyclem/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/andreasmarnold/pyclem/branch/main/graph/badge.svg)](https://codecov.io/gh/andreasmarnold/pyclem)

This package contains tools for ai-supported segmentation of platinum replica electron microscopy (PREM) images.
It further provides napari-widgets for the manual correction of the segmentation results and some useful scripts for
the (statistical) analysis of the segmentation results.

## Requirements
Make sure to install all required tools before proceeding with the installation of this package.

1. You need GIT installed on your machine, to manage the source code of this library.<br/>
   NHLBI users can install GIT from the NHLBI self-service portal. Otherwise, you can download GIT from the official website:
    - https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
2. You need to have Conda installed on your machine (to manage Python environments). This can be any conda distribution,
   but I would recommend a light-weight distribution such as miniconda.<br/>
   **Note: Install for your user only, not for all users!** This avoids needing admin rights for package installations.
    - https://docs.anaconda.com/free/miniconda/index.html
3. You'll need to install CUDA 12.4.1 (newer versions should work but haven't been tested) to make use of the
   GPU-accelerated segmentation tools. **NHLBI users will need IT-support for this!**
    - https://developer.nvidia.com/cuda-toolkit-archive
4. You need to add the CUDA neural network library cudnn 9.1.1.17 for CUDA 12 to your CUDA installation.
    - NHLBI users can find the library on the L-drive:<br/>
      L:\Lab-Taraska\AA_Group_Projects\AA_Deeplearning\cudnn-windows-x86_64-9.1.1.17_cuda12-archive
    - General users can download the library from the official website:<br/>
      https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/
    - Copy all files from the downloaded library folder to the CUDA 12.4 driver directory.<br/> 
      **NHLBI users will need IT-support for this!**
5. On Windows, you must have the Visual C++ 2015 build tools on your path! <br/>
   NHLBI users will need IT-support for installation.
    - Install them from https://visualstudio.microsoft.com/visual-cpp-build-tools/
      
6. **Optional:** If you plan on changing the source code, I recommend using an intelligent developing environment (IDE)
   such as PyCharm or Visual Studio Code. Both are free and provide excellent support for Python development.
    - PyCharm: https://www.jetbrains.com/pycharm/download/
    - Visual Studio Code: https://code.visualstudio.com/download

# Installation
1. In git-bash:
   - Clone the pyclem repository to a local hard-drive. To do so, open git bash in the desired location and run
    ```
    git clone https://git.lobos.nih.gov/taraskalab/pyclem.git
    ```

2. In Conda-prompt:
   - Create a new virtual environment with Python>=3.10
    ```ruby
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
    python -m pip install .
    ```
   - Install pycoco tools separately from the official GitHub repository
    ```
    python -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
   ```
## Update
1. In git-bash:
   - Navigate to the folder containing your clone of the PyCLEM repository.
    ```
    cd <your_folder_structure>/pyclem
    ```
   - Pull the latest changes from the repository
    ```
    git pull
    ```
   - In case you changed anything in the source code, you might need to either merge delete your changes before pulling.<br/>
     Check online for instructions on how to do this.
2. In Conda-prompt:
   - Activate the PyCLEM environment
    ```
    conda activate pyclem-env
    ```
    - Navigate to the folder containing your clone of the PyCLEM repository.
    ```
    cd <your_folder_structure>/pyclem
    ```
   - Reinstall the project with
    ```
    python -m pip install .
    ```
   
## Usage
1. Copy (and rename) the analysis template from ../pyclem/examples/analysis_template.ipynb to your data folder.

2. Open Anaconda prompt and navigate to your data folder with
```
cd A:/Path/to/your/data
```
3. Activate your PYCLEM environment and start a jupyter instance
```
conda activate pyclem-env
```
```
jupyter notebook
```
4. This will open a Jupyter application in your web browser. You can now open the jupyter notebook analysis_template.ipynb and follow the instructions in the notebook.



