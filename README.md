# beating-atari
Modularized implementation of popular deep RL algorithms in PyTorch to beat Atari 2600 games. Easy switch between algorithms and challenging games.

Implemented algorithms:
 - [x] Nips DQN [[1]](#references)
 - [x] Nature DQN [[2]](#references)
 - [x] Double DQN [[3]](#references)
 - [ ] Prioritised Experience Replay [[4]](#references)
 - [ ] Dueling Network Architecture [[5]](#references)

## Requirement
 - gym
 - PyTorch
 - OpenCV
 - tensorboard

This project runs on python >= 3.6, use pip to install dependencies:
```
pip3 install -r requirements.txt
```

### Project report
See project report [here](https://www.researchgate.net/publication/335392857_The_genesis_of_beating_Atari_games).

References
----------

[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)  
[3] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[4] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[5] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)