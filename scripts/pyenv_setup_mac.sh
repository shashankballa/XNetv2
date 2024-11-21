rm -rf pyenv
python3.10 -m venv pyenv 
source pyenv/bin/activate
pip install --upgrade pip
pip cache purge
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir
pip3 install -r requirements.txt
python -c "import torch; print('Torch version:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"