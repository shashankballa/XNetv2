if [[ "$*" == *"--rebuild"* ]]; then
    echo -e "Rebuilding the environment"
    pyenv/bin/pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    pyenv/bin/pip3 install -r requirements.txt
else
    rm -rf pyenv
    python3.10 -m venv pyenv 
    pyenv/bin/pip3 install --upgrade pip
    pyenv/bin/pip3 cache purge
    pyenv/bin/pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir
    pyenv/bin/pip3 install -r requirements.txt --no-cache-dir
fi
pyenv/bin/python -c "import torch; print('Torch version:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"