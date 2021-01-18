Works on:
- Ubuntu 20.04
- Python 3.8.5
- OpenAI gym 0.17.3
- mujoco 200
- mujoco_py 2.0.2.13
- nvidia driver 450 / 460
- CUDA 10.1
- Tensorflow 2.3.1

# Steps to reproduce my experiments:
- Install above dependencies
- Pull this repository
- Run all cells in the AlgEvo.ipynb notebook

[Hackmd for tracking progress](https://hackmd.io/NUtcNODST3SHuBCEYfpFUg?view)

# InvertedPendulum-v2
![](imgs/InvertedPendulum-v2.png)

## DDPG Algorithm
![](imgs/DDPG/InvertedPendulum-v2/train_reward_test2-pt1.jpg)
![](imgs/DDPG/InvertedPendulum-v2/test_reward_test2-pt1.jpg)
![](imgs/DDPG/InvertedPendulum-v2/train_reward_test2-pt2.jpg)
![](imgs/DDPG/InvertedPendulum-v2/test_reward_test2-pt2.jpg)

## ERL Algorithm (Train reward 10x too high)
![](imgs/ERL/InvertedPendulum-v2/train_reward_test1-pt1.jpg)
![](imgs/ERL/InvertedPendulum-v2/test_reward_test1-pt1.jpg)
![](imgs/ERL/InvertedPendulum-v2/train_reward_test1-pt2.jpg)
![](imgs/ERL/InvertedPendulum-v2/test_reward_test1-pt2.jpg)
![](imgs/ERL/InvertedPendulum-v2/train_reward_test1-pt3.jpg)
![](imgs/ERL/InvertedPendulum-v2/test_reward_test1-pt3.jpg)
![](imgs/ERL/InvertedPendulum-v2/train_reward_test2.jpg)
![](imgs/ERL/InvertedPendulum-v2/test_reward_test2.jpg)
![](imgs/ERL/InvertedPendulum-v2/actors_reward_test2.jpg)


## Previous experiments (DDPG)

![](imgs/prev/train_reward1.jpg)
![](imgs/prev/train_reward2.jpg)
![](imgs/prev/train_reward3.jpg)
![](imgs/prev/train_reward4.jpg)


### 4x times more epochs, action noise scaled logarythmically
### 0.1 -> 0.001:
![](imgs/prev/train_reward_lognoise_1.jpg)

### 0.001 -> 0.0001
![](imgs/prev/train_reward_lognoise_2.jpg)

### 0.1 -> 0.0001, 20000 epochs
![](imgs/prev/train_reward_lognoise_long_run.jpg)

## Test reward plots:

![](imgs/prev/test_reward1.jpg)
![](imgs/prev/test_reward2.jpg)

### 4x times more epochs, action noise scaled logarythmically
### 0.1 -> 0.001:
![](imgs/prev/test_reward_lognoise_1.jpg)

### 0.001 -> 0.0001
![](imgs/prev/test_reward_lognoise_2.jpg)

### 0.1 -> 0.0001, 20000 epochs
![](imgs/prev/test_reward_lognoise_long_run.jpg)

One test run had 2000 steps
