ARG1=$1
shift
find . -name "._*" -type f -delete
find . -name ".DS_*" -type f -delete
find . -name "__pycache__" -type f -delete

source .pyenv/bin/activate
if [ $? -eq 0 ]; then
    .pyenv/bin/python "$ARG1" "$@"
    deactivate
else
    echo "Failed to activate virtual environment"
    exit 1
fi

# Clear all MAC redundant files
find . -name "._*" -type f -delete
find . -name ".DS_*" -type f -delete
find . -name "__pycache__" -type f -delete