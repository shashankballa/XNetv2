ARG1=$1
shift

pyenv/bin/python "$ARG1" "$@"

# source pyenv/bin/activate
# if [ $? -eq 0 ]; then
#     deactivate
# else
#     echo "Failed to activate virtual environment"
#     exit 1
# fi