Works on:
- Ubuntu 20.04
- Python 3.8.5
- OpenAI gym 0.17.3
- mujoco 200
- mujoco_py 2.0.2.13
- nvidia driver 450
- CUDA 10.1
- Tensorflow 2.3.1

[Hackmd for tracking progress](https://hackmd.io/NUtcNODST3SHuBCEYfpFUg?view)

# DDPG Algorithm
## InvertedPendulum-v2
## Training reward plots:

![](imgs/train_reward1.jpg)
![](imgs/train_reward2.jpg)
![](imgs/train_reward3.jpg)
![](imgs/train_reward4.jpg)


### 4x times more epochs, action noise scaled logarythmically
### 0.1 -> 0.001:
![](imgs/train_reward_lognoise_1.jpg)

### 0.001 -> 0.0001
![](imgs/train_reward_lognoise_2.jpg)

## Test reward plots:

![](imgs/test_reward1.jpg)
![](imgs/test_reward2.jpg)

### 4x times more epochs, action noise scaled logarythmically
### 0.1 -> 0.001:
![](imgs/test_reward_lognoise_1.jpg)

### 0.001 -> 0.0001
![](imgs/test_reward_lognoise_2.jpg)

One test run had 2000 steps
