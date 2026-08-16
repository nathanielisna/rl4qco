## Description
This repository contains the main implementation for the paper:
"Comparative Analysis of Actor-Critic Reinforcement Learning Algorithms for Heuristic-Based Quantum Circuit Optimization".

This repository has 3 main components: There is a dataset generator `./source/model/QuantumCircuitDataset.py`, a reinforcement learning environment `./source/model/QuantCircOptEnv.py`, and a CNN agent `./source/model/Policy.py`.

## Usage of provided code
Two example training scripts `./train/train_scratch.py` and `./train/train_pretrained_critic.py` have been provided to demonstrate usage of dataset generator, reinforcement learning environment, and CNN agent. The first trains a reinforcement learning model from scratch. The second first loads a pretrained value function, then trains the rest of the reinforcement learning agent.

To train, run the shell script provided `./train/train.sh` from the root directory as such:
```
bash train/train.sh
```
The shell script will require [tmux](https://github.com/tmux/tmux/wiki/Getting-Started) (installed in most Linux distributions) to keep the training script active. If the session detaches, you can re-attach using `tmux attach -t train`.

## Additional libraries
The dataset generator and the reinforcement learning environment relies on the quantum circuit transformation library [rl4circopt](https://github.com/google-research/google-research/tree/master/rl4circopt) from Google. A compatible version of rl4circopt is included in `./source/lib/rl4circopt` with the following changes.

1. **Additional rule "TripleCnot"** class has been implemented into the `rl4circopt` library:  
   https://github.com/google-research/google-research/tree/master/rl4circopt  
   - File: `rules.py`  
   - Line: 274 