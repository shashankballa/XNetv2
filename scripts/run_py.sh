ARG1=$1
shift

source .pyenv/bin/activate
if [ $? -eq 0 ]; then
    .pyenv/bin/python "$ARG1" "$@"
    deactivate
else
    echo "Failed to activate virtual environment"
    exit 1
fi