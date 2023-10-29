# Setup everything for this repository

readonly ARGS="$@"  # Reset using https://stackoverflow.com/a/4827707
readonly PROGNAME=$(basename $0)
readonly PROGPATH=$(realpath $(dirname $0))

# Internal variables
env_name="featmf"   # Name of the environment
exec_name="conda"           # Executable
dry_run="false"     # 'true' or 'false'
ask_prompts="true"  # 'true' or 'false'
dev_tools="false"   # 'true' or 'false'
warn_exit="true"    # 'true' or 'false'

# Output formatting
debug_msg_fmt="\e[2;90m"
info_msg_fmt="\e[1;37m"
warn_msg_fmt="\e[1;35m"
fatal_msg_fmt="\e[2;31m"
command_msg_fmt="\e[0;36m"
# Wrapper printing functions
echo_debug () {
    echo -ne $debug_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
echo_info () {
    echo -ne $info_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
echo_warn () {
    echo -ne $warn_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
echo_fatal () {
    echo -ne $fatal_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
echo_command () {
    echo -ne $command_msg_fmt
    echo $@
    echo -ne "\e[0m"
}
# Installer functions
function run_command() {
    echo_command $@
    if [ $dry_run == "true" ]; then
        echo_debug "Dry run, not running command..."
    else
        $@
    fi
}
function conda_install() {
    run_command $exec_name install -y --freeze-installed --no-update-deps $@
    ec=$?
    if [[ $ec -gt 0 ]]; then
        echo_warn "Could not install '$@', maybe try though conda_raw_install"
        if [[ $warn_exit == "true" ]]; then
            exit $ec
        else
            echo_debug "Exit on warning not set, continuing..."
        fi
    fi
}
function conda_raw_install() {
    run_command $exec_name install -y $@
}
function pip_install() {
    run_command pip install --upgrade $@
}

# Ensure installation can happen
if [ -x "$(command -v mamba)" ]; then   # If mamba found
    echo_debug "Found mamba"
    exec_name="mamba"
elif [ -x "$(command -v conda)" ]; then # If conda found
    echo_debug "Found conda (couldn't find mamba)"
    exec_name="conda"
else
    echo_fatal "Could not find mamba or conda! Install, source, and \
            activate it."
    exit 127
fi

function usage() {
    cat <<-EOF

Environment setup for FeatMF

Usage: 
    1. bash $PROGNAME [-OPTARG VAL ...]
    2. bash $PROGNAME --help
    3. bash $PROGNAME NAME [-OPTARG VAL ...]

All optional arguments:
    -c | --conda INST       Conda installation ('mamba' or 'conda'). By
                            default, 'mamba' is used (if installed), else
                            'conda'.
    -d | --dev              If passed, the documentation and packaging 
                            tools are also installed (they aren't, by 
                            default).
        --dry-run           If passed, the commands are printed instead of
                            running them.
    -e | --env-name NAME    Name of the conda/mamba environment. This can
                            also be passed as the 1st positional argument.
    -h | --help             Show this message.
        --no-exit-on-warn   By default, a warning causes the script to
                            exit (with a suggested modification). If this
                            option is passed, the script doesn't exit (it
                            continues). This could cause unintended errors.
    -n | --no-prompt        By default, a prompt is shown (asking to press
                            Enter to continue). If this is passed, the
                            prompt is not shown.

Exit codes
    0       Script executed successfully
    1       Argument error (some wrong argument was passed)
    127     Could not find conda or mamba (executable)
EOF
}

function parse_options() {
    # Set passed arguments
    set -- $ARGS
    pos=1
    while (( "$#" )); do
        arg=$1
        shift
        case "$arg" in
            # Conda installation to use
            "--conda" | "-c")
                ci=$1
                shift
                echo_debug "Using $ci (for anaconda base)"
                exec_name=$ci
                ;;
            # Developer install options
            "--dev" | "-d")
                echo_debug "Installing documentation and packaging tools"
                dev_tools="true"
                ;;
            # Dry run
            "--dry-run")
                echo_debug "Dry run mode enabled"
                dry_run="true"
                ;;
            # Environment name
            "--env-name" | "-e")
                en=$1
                shift
                echo_debug "Using environment $en"
                env_name=$en
                ;;
            # Help options
            "--help" | "-h")
                usage
                exit 0
                ;;
            # No exit on warning
            "--no-exit-on-warn")
                echo_debug "No exit on warning set"
                warn_exit="false"
                ;;
            # No prompt
            "--no-prompt" | "-n")
                echo_debug "Not showing prompts (no Enter needed)"
                ask_prompts="false"
                ;;
            *)
                if [ $pos -eq 1 ]; then # Environment name
                    echo_debug "Using environment $arg"
                    env_name=$arg
                else
                    echo_fatal "Unrecognized option: $arg"
                    echo_debug "Run 'bash $PROGNAME --help' for usage"
                    exit 1
                fi
        esac
        pos=$((pos + 1))
    done
}

# ====== Main program entrypoint ======
parse_options
if [ -x "$(command -v $exec_name)" ]; then
    echo_info "Using $exec_name (for base anaconda)"
else
    echo_fatal "Could not find $exec_name! Install, source, and \
            activate it."
    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" != "$env_name" ]; then
    echo_fatal "Wrong environment activated. Activate $env_name"
    exit 1
fi

# Confirm environment
echo_info "Using environment: $CONDA_DEFAULT_ENV"
echo_info "Python: $(which python)"
echo_debug "Python version: $(python --version)"
echo_info "Pip: $(which pip)"
echo_debug "Pip version: $(pip --version)"
if [ $ask_prompts == "true" ]; then
    read -p "Continue? [Ctrl-C to exit, enter to continue] "
elif [ $ask_prompts == "false" ]; then
    echo_info "Continuing..."
fi

# Install packages
start_time=$(date)
start_time_secs=$SECONDS
echo_debug "---- Start time: $start_time ----"
if [ $dev_tools == "true" ]; then 
    echo_info "------ Installing documentation and packaging tools ------"
    # Sphinx docs
    conda_install -c conda-forge sphinx sphinx-rtd-theme sphinx-copybutton
    pip_install sphinx-reload
    conda_install -c conda-forge setuptools
    pip_install build
    conda_install -c conda-forge hatch hatchling twine
    conda_install -c conda-forge keyrings.alt
    conda_install conda-build anaconda-client
    conda_install -c conda-forge sphinx-design
elif [ $dev_tools == "false" ]; then
    echo_info "Skipping documentation and packaging tools"
fi
echo_info "------ Installing core packages ------"
conda_raw_install pytorch torchvision torchaudio pytorch-cuda=11.8 \
        -c pytorch -c nvidia
conda_raw_install -c conda-forge opencv
conda_install -c conda-forge joblib
conda_install -c conda-forge matplotlib
conda_install -c conda-forge jupyter
conda_install -c conda-forge pillow
conda_install -c conda-forge cupy
conda_install -c conda-forge pynvml
conda_install -c pytorch faiss-gpu
conda_install -c conda-forge einops

# Installation ended
end_time=$(date)
end_time_secs=$SECONDS
echo_debug "---- End time: $end_time ----"
# dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
dur=$(( $end_time_secs - $start_time_secs ))
_d=$(( dur/3600/24 ))   # Days!
echo_info "---- Environment setup took (d-HH:MM:SS): \
        $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo_info "----- Environment $CONDA_DEFAULT_ENV has been setup -----"
echo_debug "Starting time: $start_time"
echo_debug "Ending time: $end_time"
