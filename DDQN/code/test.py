import numpy as np
import gym
from utils import *
from agent import *
from config import *
from train_ram import train

if __name__ == '__main__':
    env = gym.make(RAM_ENV_NAME)
    all_rewards = [None for _ in range(3)]
    for i in range(3):
        print('Config {}, {}, {}. Episode {}/10'.format(DUEL, DOUBLE, PRIORITIZED, i+1))
        agent = Agent(env.observation_space.shape[0], env.action_space.n, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE, False, DUEL, DOUBLE, PRIORITIZED)
        _, all_rewards[i] = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    np.save('./lunarlander_results/{}_{}_{}_{}_rewards.npy'.format(RAM_ENV_NAME, DUEL, DOUBLE, PRIORITIZED), np.array(all_rewards))