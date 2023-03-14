# Install conda environment for development of FeatMF

_CONDA_ENV_NAME="featmf-work"

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
conda install -y -c conda-forge hatch setuptools
pip install --upgrade build
conda install -y -c conda-forge twine
conda install -y conda-build
