# MNIST-AI

# Setup (Cuda)
pip3 install -r requirements.txt

# Setup (Others)
https://pytorch.org/get-started/locally/

# Train
python main.py train {epochs}

# Test (Own Images)
# Image must be 28x28 Pixel with a black background and a white text
mkdir image <- Save image in this folder
python main.py test