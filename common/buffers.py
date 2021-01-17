from collections import namedtuple
import numpy as np
import torch


class Episode(object):
    def __init__(self, Transition):
        self.device = 'cpu'
        self.episode = []
        self.Transition = Transition

    def push(self, *args):
        self.episode.append(self.Transition(*args))

    def get(self):
        return self.Transition(*zip(*self.episode))

    def __len__(self):
        return len(self.episode)

    def clear(self):
        self.episode = []


class Memory(object):
    def __init__(self, Transition):
        self.memory = []
        self.num_episode = 0
        self.Transition = Transition

    def add(self, *args):
        self.memory.append(self.Transition(*args))

    def push(self, epi: Episode):
        self.memory += epi.episode
        self.num_episode += 1

    def sample(self):
        return self.Transition(*zip(*self.memory))

    @property
    def sample_size(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)

    def sample_clear(self):
        self.memory = []
        self.num_episode = 0

    def clear(self):
        self.memory = []
        self.num_episode = 0


class ReplayBuffer(object):

    def __init__(self, size, Transition):
        self.memory = [None for i in range(size)]
        self.ptr, self.size, self.max_size = 0, 0, size
        self.num_episode, self.sample_size = 0, 0
        self.device = 'cpu'
        self.Transition = Transition

    def add(self, *args):
        self.memory[self.ptr] = self.Transition(*args)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.sample_size += 1
        if self.memory[self.ptr].done == True:
            self.num_episode += 1

    def push(self, epi: Episode):
        if self.ptr + len(epi.episode) > self.max_size:
            self.memory[self.ptr:self.max_size] = epi.episode[0:self.max_size - self.ptr]
            self.memory[0:len(epi.episode) - (self.max_size - self.ptr)] = epi.episode[self.max_size - self.ptr:]
        else:
            self.memory[self.ptr:self.ptr + len(epi.episode)] = epi.episode

        self.ptr = (self.ptr + len(epi.episode)) % self.max_size
        self.size = min(self.size + len(epi.episode), self.max_size)
        self.sample_size += len(epi.episode)
        self.num_episode += 1

    def sample(self, idxs):
        return self.Transition(*zip(*[self.memory[idx] for idx in idxs]))

    # def random_sample(self, minibatch_size, replace=False):
    #     idxs = np.random.choice(self.size, minibatch_size, replace=replace)
    #     return self.Transition(*zip(*[self.memory[idx] for idx in idxs]))

    def __len__(self):
        return len(self.memory)

    def sample_clear(self):
        self.sample_size = 0
        self.num_episode = 0

    def clear(self):
        self.memory.clear()
        self.ptr, self.size = 0, 0
        self.num_episode, self.sample_size = 0, 0


Transition = namedtuple('PPO_Transition', ('obs', 'action', 'reward', 'done', 'value', 'logproba'))


class BufferPPO2(object):

    def __init__(self, device, gamma=0.99, lamda=0.97, advantage_norm=True):
        self.memory = []
        self.device = device
        self.gamma = gamma
        self.lamda = lamda
        self.advantage_norm = advantage_norm
        self.EPS = 1e-10
        self.num_episode = 0

    def add(self, *args):
        self.memory.append(Transition(*args))

    def push(self, epi: Episode):
        self.memory += epi.episode
        self.num_episode += 1

    def sample(self):

        memory = Transition(*zip(*self.memory))

        obs = torch.Tensor(memory.obs).to(self.device)
        actions = torch.Tensor(memory.action).to(self.device)
        rewards = torch.Tensor(memory.reward).to(self.device)
        done = torch.Tensor(memory.done).to(self.device)
        values = torch.Tensor(memory.value).to(self.device)
        logproba = torch.Tensor(memory.logproba).to(self.device)
        returns = torch.Tensor(len(self.memory)).to(self.device)
        advantages = torch.Tensor(len(self.memory)).to(self.device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(len(self.memory))):
            mask = (1 - done[i])
            returns[i] = rewards[i] + self.gamma * prev_return * mask
            # implement GAE-Lambda advantage calculation
            deltas = rewards[i] + self.gamma * prev_value * mask - values[i]
            advantages[i] = deltas + self.gamma * self.lamda * prev_advantage * mask

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        if self.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.EPS)

        return dict(obs=obs,
                    act=actions,
                    ret=returns,
                    adv=advantages,
                    v=values,
                    logproba=logproba,)

    def __len__(self):
        return len(self.memory)

    def sample_clear(self):
        self.memory = []
        self.num_episode = 0

    def clear(self):
        self.memory = []
        self.num_episode = 0
