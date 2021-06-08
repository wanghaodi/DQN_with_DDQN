import config
from utils import SumTree

import numpy as np
import random
import bisect
import torch

ALPHA = config.alpha
EPSILON = config.epsilon
TD_INIT = config.td_init

class Replay_Buffer:
    '''
    Vanilla replay buffer
    '''
    
    def __init__(self, capacity=int(1e6), batch_size=None):
        
        self.capacity = capacity
        self.memory = [None for _ in range(capacity)] # save tuples (state, action, reward, next_state, done)
        self.ind_max = 0 # how many transitions have been stored
        
    def remember(self, state, action, reward, next_state, done):
        
        ind = self.ind_max % self.capacity
        self.memory[ind] = (state, action, reward, next_state, done)
        self.ind_max += 1
        
    def sample(self, k):
        '''
        return sampled transitions. Make sure that there are at least k transitions stored before calling this method 
        '''
        index_set = random.sample(list(range(len(self))), k)
        states = torch.from_numpy(np.vstack([self.memory[ind][0] for ind in index_set])).float()
        actions = torch.from_numpy(np.vstack([self.memory[ind][1] for ind in index_set])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[ind][2] for ind in index_set])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[ind][3] for ind in index_set])).float()
        dones = torch.from_numpy(np.vstack([self.memory[ind][4] for ind in index_set]).astype(np.uint8)).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return min(self.ind_max, self.capacity)
        
class Rank_Replay_Buffer:
    '''
    Rank-based replay buffer
    '''
    
    def __init__(self, capacity=int(1e6), batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = ALPHA
        self.memory = [None for _ in range(capacity)]
        self.segments = [-1] + [None for _ in range(batch_size)] # the ith index will be in [segments[i-1]+1, segments[i]]
        
        self.errors = [] # saves (-TD_error, index of transition), sorted
        self.memory_to_rank = [None for _ in range(capacity)]
        
        self.ind_max = 0 # how many transitions have been stored
        self.total_weights = 0 # sum of p_i
        self.cumulated_weights = []
        
    def remember(self, state, action, reward, next_state, done):
        index = self.ind_max % self.capacity
        if self.ind_max >= self.capacity: # memory is full, need to pop
            self.pop(index)
        else: # memory is not full, need to adjust weights and find segment points
            self.total_weights += (1/(1+self.ind_max))**self.alpha # memory is not full, calculate new weights
            self.cumulated_weights.append(self.total_weights)
            self.update_segments()
        
        max_error = -self.errors[0][0] if self.errors else 0
        self.insert(max_error, index)
        self.memory[index] = (state, action, reward, next_state, done)
        self.ind_max += 1
        
    def sample(self, batch_size=None): # notive that batch_size is not used. It's just to unify the calling form
        index_set = [random.randint(self.segments[i]+1, self.segments[i+1]) for i in range(self.batch_size)]
        probs = torch.from_numpy(np.vstack([(1/(1+ind))**self.alpha/self.total_weights for ind in index_set])).float()
        
        index_set = [self.errors[ind][1] for ind in index_set]
        states = torch.from_numpy(np.vstack([self.memory[ind][0] for ind in index_set])).float()
        actions = torch.from_numpy(np.vstack([self.memory[ind][1] for ind in index_set])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[ind][2] for ind in index_set])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[ind][3] for ind in index_set])).float()
        dones = torch.from_numpy(np.vstack([self.memory[ind][4] for ind in index_set]).astype(np.uint8)).float()
        for ind in index_set:
            self.pop(ind)
        
        return index_set, states, actions, rewards, next_states, dones, probs
    
    def insert(self, error, index):
        '''
        Input : 
            error : the TD-error of this transition
            index : the location of this transition
        insert error into self.errors, update self.memory_to_rank and self.rank_to_memory accordingly
        '''
        ind = bisect.bisect(self.errors, (-error, index))
        self.memory_to_rank[index] = ind
        self.errors.insert(ind, (-error, index))
        for i in range(ind+1, len(self.errors)):
            self.memory_to_rank[self.errors[i][1]] += 1
        
    def pop(self, index):
        '''
        Input :
            index : the location of a transition
        remove this transition, update self.memory_to_rank and self.rank_to_memory accordingly
        '''
        ind = self.memory_to_rank[index]
        self.memory_to_rank[index] = None
        self.errors.pop(ind)
        for i in range(ind, len(self.errors)):
            self.memory_to_rank[self.errors[i][1]] -= 1
        
    def update_segments(self):
        '''
        Update the segment points.
        '''
        if self.ind_max+1 < self.batch_size: # if there is no enough transitions
            return None
        for i in range(self.batch_size):
            ind = bisect.bisect_left(self.cumulated_weights, self.total_weights*((i+1)/self.batch_size))
            self.segments[i+1] = max(ind, self.segments[i]+1)
            
    def __len__(self):
        return min(self.capacity, self.ind_max)
    

class Proportion_Replay_Buffer:
    '''
    Proportion-based replay buffer
    '''
    
    def __init__(self, capacity=int(1e6), batch_size=None):
        self.capacity = capacity
        self.alpha = ALPHA
        self.memory = [None for _ in range(capacity)]
        self.weights = SumTree(self.capacity)
        self.default = TD_INIT
        self.ind_max = 0
        
    def remember(self, state, action, reward, next_state, done):
        index = self.ind_max % self.capacity
        self.memory[index] = (state, action, reward, next_state, done)
        delta = self.default+EPSILON - self.weights.vals[index+self.capacity-1]
        self.weights.update(delta, index)
        self.ind_max += 1
        
    def sample(self, batch_size):
        index_set = [self.weights.retrive(self.weights.vals[0]*random.random()) for _ in range(batch_size)]
        #print(index_set)
        probs = torch.from_numpy(np.vstack([self.weights.vals[ind+self.capacity-1]/self.weights.vals[0] for ind in index_set])).float()                     
        
        states = torch.from_numpy(np.vstack([self.memory[ind][0] for ind in index_set])).float()
        actions = torch.from_numpy(np.vstack([self.memory[ind][1] for ind in index_set])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[ind][2] for ind in index_set])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[ind][3] for ind in index_set])).float()
        dones = torch.from_numpy(np.vstack([self.memory[ind][4] for ind in index_set]).astype(np.uint8)).float()

        return index_set, states, actions, rewards, next_states, dones, probs
                                 
    def insert(self, error, index):
        delta = error+EPSILON - self.weights.vals[index+self.capacity-1]
        self.weights.update(delta, index)
            
    def __len__(self):
        return min(self.capacity, self.ind_max)