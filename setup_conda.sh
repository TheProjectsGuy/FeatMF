# Install conda environment for development of FeatMF

_CONDA_ENV_NAME="${1:-featmf-work}"

# Ensure conda is installed
if ! [ -x "$(command -v conda)" ]; then
    echo 'Error: conda is not installed. Source or install Anaconda'
    exit 1
fi
# Ensure environmnet
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo 'No conda environment activated'
    exit 1
fi
if [ "$CONDA_DEFAULT_ENV" != "$_CONDA_ENV_NAME" ]; then
    echo "Wrong conda environment activated. Activate $_CONDA_ENV_NAME"
    exit 1
fi

# Install everything
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
read -p "Continue? [Ctrl-C to exit, enter to continue] "

# Install requirements
echo "---- Installing documentation and packaging tools ----"
conda install -y -c conda-forge sphinx sphinx-rtd-theme sphinx-copybutton
pip install sphinx-reload
conda install -y -c conda-forge setuptools
pip install --upgrade build
conda install -y -c conda-forge hatch hatchling twine
conda install -y conda-build anaconda-client
conda install -y -c conda-forge sphinx-design
echo "---- Installing core package dependencies ----"
conda install -y -c nvidia cuda-toolkit
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c conda-forge opencv
conda install -y -c conda-forge joblib
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge jupyter
conda install -y -c conda-forge pillow

# Installation completed
echo "Environment $CONDA_DEFAULT_ENV is ready with all packages installed"
