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

# Functions
function conda_install() {
    echo -ne "\e[0;36m"
    echo "conda install -y --freeze-installed --no-update-deps $@"
    echo -ne "\e[0m"
    conda install -y --freeze-installed --no-update-deps $@
}
function pip_install() {
    echo -ne "\e[0;36m"
    echo "pip install --upgrade $@"
    echo -ne "\e[0m"
    pip install --upgrade $@
}

# Install requirements
echo "---- Installing documentation and packaging tools ----"
conda_install -c conda-forge sphinx sphinx-rtd-theme sphinx-copybutton
pip_install sphinx-reload
conda_install -c conda-forge setuptools
pip_install build
conda_install -c conda-forge hatch hatchling twine
conda_install conda-build anaconda-client
conda_install -c conda-forge sphinx-design
echo "---- Installing core package dependencies ----"
conda_install -c nvidia cuda-toolkit
conda_install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda_install -c conda-forge opencv
conda_install -c conda-forge joblib
conda_install -c conda-forge matplotlib
conda_install -c conda-forge jupyter
conda_install -c conda-forge pillow

# Installation completed
echo "Environment $CONDA_DEFAULT_ENV is ready with all packages installed"
