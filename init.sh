# !/bin/bash

python -m venv .venv && \
source .venv/bin/activate.fish && \
pip install -r requierments.txt && \
echo "Sucessfully create a venv and install every necessary packages"