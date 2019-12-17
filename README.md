# Master's Thesis
- Multi-task Reinforcement Learning in Python based on Stable baselines A2C implementation.
- Framework: Tensorflow
- Learning environment: Open AI gym
### Theory is based on:
- https://arxiv.org/abs/1702.06053

# Installation
### Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You’ll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

Also you'll need to install the requirements from requirements.txt
```bash
pip install -r requirements.txt
```

# Usage
You can use simply the main.py
```bash
python main.py
```
For more information see
```bash
python main.py -h
```
