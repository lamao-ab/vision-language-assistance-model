pip3 install torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip3 install nvidia-cudss-cu12 --no-deps

echo 'export LD_LIBRARY_PATH=$(dirname $(find ~/.local -name libcudss.so.0 | head -n 1)):$LD_LIBRARY_PATH' >> ~/.bashrc


sudo python3 -c "import torch; print('CUDA Ready:', torch.cuda.is_available())"


sudo apt-get install -y portaudio19-dev espeak

pip3 install  "numpy==1.26.4" "huggingface_hub>=0.23.0" "transformers>=4.44.0" "accelerate>=0.34.0" "peft>=0.13.0" "scipy>=1.11.0" "pillow>=10.0.0" "bitsandbytes>=0.43.0" opencv-python-headless SpeechRecognition pyttsx3 PyAudio




Set up bleaotttooth

set up HF:
python3 -c "from huggingface_hub import login; login()"

Set up memory
The Fix: "Download More RAM" (Creating a Swap File)
We are going to carve out 8GB of space on your Jetson's SSD/SD Card and tell Linux to use it as an emergency "overflow" RAM. This will give Hugging Face the breathing room it needs to unpack the model and compress it into VRAM without crashing.

Run these commands one by one in your terminal (we must use sudo here because we are modifying the core Linux memory manager):

1. Create an 8 Gigabyte file:

Bash
sudo fallocate -l 8G /swapfile
2. Lock the file down so only the system can use it:

Bash
sudo chmod 600 /swapfile
3. Format it as Swap memory:

Bash
sudo mkswap /swapfile
4. Turn the Swap memory on:

Bash
sudo swapon /swapfile
(Optional but highly recommended): Make this permanent so the Jetson remembers it after a reboot:

Bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
The Next Step
Now that your Jetson effectively has 16GB of working memory (8GB physical + 8GB swap), the loading process will succeed.

Run your assistant again:

Bash
python3 blind-assist-final.py




# 1. Point the system to Jetson's hidden CUDA tools
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 2. Download the raw bitsandbytes source code
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes

# 3. Compile the C++ and CUDA code for your specific hardware
cmake -DCOMPUTE_BACKEND=cuda -S .
make

# 4. Install your newly built custom library
pip3 install .



