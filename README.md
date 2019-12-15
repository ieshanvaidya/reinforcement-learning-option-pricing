# reinforcement-learning-option-pricing
A DQN agent that optimally hedges an options portfolio.

## Usage
```
usage: agent.py [-h] [--nepisodes EPISODES] [--episode_length LENGTH] [--epsilon EPSILON]
               [--decay DECAY] [--epsilon_min EPSILON_MIN] [--gamma GAMMA] [--update_every STEPS]
               [--checkpoint_every STEPS] [--resume] [--batch_size BATCH_SIZE]
               [--replay_memory_size SIZE] [--seed SEED] [--savedir SAVEDIR] [--nhidden NHIDDEN]
               [--nunits NUNITS] [--lr LR] [--beta1 BETA1] [--cuda] [--scale SCALE [--clip CLIP]
               [--best_reward_criteria CRITERIA] [--trc_multiplier MULTIPLIER] [--trc_ticksize TICKSIZE]

general arguments:
  -h, --help                        show this help message and exit
  --resume                          resume training from checkpoint, requires existing savedir folder
  --savedir SAVEDIR                 save directory
  --seed SEED                       seed for RNG
  --cuda                            enables cuda
  --batch_size BATCH_SIZE           batch size, default=128
               
agent arguments:
  --nepisodes EPISODES              number of episodes to train, default=15000
  --episode_length LENGTH           maximum episode length, default=1000
  --epsilon EPSILON                 starting e-greedy probability, default=1
  --decay DECAY                     decay factor multiplied to epsilon after each episode, default=0.999
  --epsilon_min EPSILON_MIN         minumum value taken by epsilon, default=0.005
  --gamma GAMMA                     discount factor, default=0.3
  --update_every STEPS              update target model every [_] steps, default=500
  --checkpoint_every STEPS          checkpoint model every [_] steps, default=1000
  --replay_memory_size SIZE         replay memory size, default=64000
  --nhidden NHIDDEN                 number of hidden layers, default=2
  --nunits NUNITS                   number of units in a hidden layer, default=128
  --lr LR                           learning rate, default=0.001
  --beta1 BETA1                     beta1 for adam. default=0.9
  --scale SCALE                     scale reward by [_] | reward = [_] * reward, takes priority over clip, default=1
  --clip CLIP                       clip reward between [-clip, clip], pass in 0 for no clipping, default=100
  --best_reward_criteria CRITETIA   save model if mean reward over last [_] episodes greater than best reward, default=10
  
environment arguments:
  --trc_multiplier MULTIPLIER       transaction cost multiplier, default=1
  --trc_ticksize TICKSIZE           transaction cost ticksize, default=0.1
```
