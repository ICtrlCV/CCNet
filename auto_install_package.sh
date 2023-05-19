#!/bin/bash
# Install regular dependencies
pip3 install coloredlogs==15.0.1 matplotlib==3.4.3 numpy==1.23.0 opencv_python==4.5.5.64 Pillow==9.2.0 pycocotools==2.0.5 tensorboardX==2.5.1 tqdm==4.64.0

# Check if there is a graphics card
if lspci | grep -i NVIDIA > /dev/null ; then
    pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
else
    pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
fi
