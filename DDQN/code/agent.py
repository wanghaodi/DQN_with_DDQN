import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np

from networks import *
from ReplayBuffer import Replay_Buffer, Rank_Replay_Buffer, Proportion_Replay_Buffer

class Agent:

    def __init__(self, state_size, action_size, bs, lr, tau, gamma, device, visual=False, duel=False, double=False, prioritized=False):
        '''
        When dealing with visual inputs, state_size should work as num_of_frame
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.double = double
        self.prioritized = prioritized
        if visual:
            self.Q_local = Visual_Q_Network(self.state_size, self.action_size, duel=duel).to(self.device)
            self.Q_target = Visual_Q_Network(self.state_size, self.action_size, duel=duel).to(self.device)
        else:
            self.Q_local = Q_Network(self.state_size, self.action_size, duel=duel).to(device)
            self.Q_target = Q_Network(self.state_size, self.action_size, duel=duel).to(device)
        self.soft_update(1)
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)
        if not self.prioritized:
            self.memory = Replay_Buffer(int(1e5), bs)
        else:
            #self.memory = Rank_Replay_Buffer(int(1e5), bs)
            # or
            self.memory = Proportion_Replay_Buffer(int(1e5), bs)

    def act(self, state, eps=0):
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_values = self.Q_local(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        if not self.prioritized:
            states, actions, rewards, next_states, dones = self.memory.sample(self.bs)
            w = torch.ones(actions.size())
            w = w.to(self.device)
        else:
            index_set, states, actions, rewards, next_states, dones, probs = self.memory.sample(self.bs)
            w = 1/len(self.memory)/probs
            w = w/torch.max(w)
            w = w.to(self.device)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        Q_values = self.Q_local(states)
        Q_values = torch.gather(input=Q_values, dim=-1, index=actions)

        with torch.no_grad():
            Q_targets = self.Q_target(next_states)
            if not self.double:
                Q_targets, _ = torch.max(input=Q_targets, dim=1, keepdim=True)
            else:
                inner_actions = torch.max(input=self.Q_local(next_states), dim=1, keepdim=True)[1]
                Q_targets = torch.gather(input=Q_targets, dim=1, index=inner_actions)
            Q_targets = rewards + self.gamma * (1 - dones) * Q_targets

        deltas = Q_values - Q_targets
        loss = (w*deltas.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.prioritized:
            deltas = np.abs(deltas.detach().cpu().numpy().reshape(-1))
            for i in range(self.bs):
                self.memory.insert(deltas[i], index_set[i])

    def soft_update(self, tau):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)