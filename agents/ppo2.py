import numpy as np
import random
import torch.optim as optim
from common.utils import *
from common.networks import *
import math

EPS = 1e-10


class Agent():

    def __init__(self,
                 # config
                 obs_dim,
                 act_dim,
                 config,

                 # algo parameter
                 minibatch_size=256,
                 num_epoch=10,
                 hidden_sizes=(64, 64),
                 gamma=0.995,
                 lamda=0.97,
                 lr=3e-4,
                 ratio_clip=0.2,
                 loss_coeff_value=0.5,
                 loss_coeff_entropy=0.01,
                 schedule_clip='linear',
                 schedule_adam='linear',

                 layer_norm=True,
                 state_norm=False,
                 lossvalue_norm=True,
                 advantage_norm=True,

                 ):

        self.config = config
        if config.gpu_index == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device(
                'cpu')

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.gamma = self.config.agent['gamma'] if 'gamma' in self.config.agent.keys() else gamma
        self.lamda = self.config.agent['lamda'] if 'lamda' in self.config.agent.keys() else lamda
        self.lr = self.config.agent['lr'] if 'lr' in self.config.agent.keys() else lr
        self.num_epoch = self.config.agent['num_epoch'] if 'num_epoch' in self.config.agent.keys() else num_epoch
        self.minibatch_size = self.config.agent[
            'minibatch_size'] if 'minibatch_size' in self.config.agent.keys() else minibatch_size
        self.ratio_clip = self.config.agent['ratio_clip'] if 'ratio_clip' in self.config.agent.keys() else ratio_clip
        self.hidden_sizes = hidden_sizes
        self.loss_coeff_value = self.config.agent[
            'loss_coeff_value'] if 'loss_coeff_value' in self.config.agent.keys() else loss_coeff_value
        self.loss_coeff_entropy = self.config.agent[
            'loss_coeff_entropy'] if 'loss_coeff_entropy' in self.config.agent.keys() else loss_coeff_entropy

        self.schedule_clip = schedule_clip
        self.schedule_adam = schedule_adam
        self.lossvalue_norm = lossvalue_norm
        self.advantage_norm = advantage_norm
        self.layer_norm = layer_norm
        self.state_norm = state_norm

        self.record_policy_loss = 0
        self.record_vf_loss = 0
        self.record_entropy_loss = 0

        # Main network
        self.policy = PPOModel(self.obs_dim, self.act_dim, use_layer_norm=layer_norm).to(self.device)
        # Create optimizers
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def set_policy(self, policy):
        self.policy = policy

    # def select_action(self, obs):
    #     if type(obs) is np.ndarray:
    #         obs = torch.Tensor(obs).to(self.device)
    #     action_mean, action_logstd, value = self.policy(obs.unsqueeze(0))
    #     action_std = torch.exp(action_logstd)
    #     action = torch.normal(action_mean, action_std)
    #     logproba = normal_logproba(action, action_mean, action_logstd, action_std)
    #     return action, logproba, value, action_mean, action_std

    # samples: Transition(*zip(*self.memory))
    def train_model(self, samples, iter_index):
        if not self.config.sampler_gpu_index == self.config.agent_gpu_index:
            self.policy.to(self.device)

        batch_size = len(samples.obs)
        samples = self.process_return(samples)

        self.record_policy_loss = 0
        self.record_vf_loss = 0
        self.record_entropy_loss = 0

        sample_batch = range(batch_size)
        update_times = int(self.num_epoch * batch_size / self.minibatch_size)
        for i_epoch in range(update_times):
            # choice方法会改变随机种子
            # batch_index =np.random.choice(sample_batch, self.minibatch_size,replace=False)
            batch_index = random.sample(sample_batch, self.minibatch_size)
            obs = samples['obs'][batch_index]
            act = samples['act'][batch_index]
            ret = samples['ret'][batch_index]
            adv = samples['adv'][batch_index]
            oldlogproba = samples['logproba'][batch_index]
            action_mean, action_logstd = self.policy.forward_actor(obs)
            newlogproba = normal_logproba(act, action_mean, action_logstd)
            new_value = self.policy.forward_critic(obs).flatten()
            ratio = torch.exp(newlogproba - oldlogproba)
            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip) * adv
            loss_surr = - torch.mean(torch.min(surr1, surr2))
            self.record_policy_loss += loss_surr.item()

            loss_value = torch.mean((new_value - ret).pow(2))
            self.record_vf_loss += loss_value.item()
            if self.lossvalue_norm:
                loss_value = loss_value / ret.std()

            loss_entropy = torch.mean(torch.exp(newlogproba) * newlogproba)
            # loss_entropy = -torch.mean(0.5 + 0.5 * math.log(2 * math.pi) + action_logstd)
            self.record_entropy_loss += loss_entropy.item()

            total_loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # debug
            # print('loss:', total_loss, "=",
            # loss_surr, "+", self.loss_coeff_value, "*", loss_value, "+", self.loss_coeff_entropy, "*", loss_entropy)

            # Save losses

        if self.schedule_clip == 'linear':
            ep_ratio = 1 - (iter_index / self.config.iterations)
            self.ratio_clip = self.ratio_clip * ep_ratio

        if self.schedule_adam == 'linear':
            ep_ratio = 1 - (iter_index / self.config.iterations)
            lr = self.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        self.record_policy_loss = self.record_policy_loss / update_times
        self.record_entropy_loss = self.record_entropy_loss / update_times
        self.record_vf_loss = self.record_vf_loss / update_times

        # samples: Transition(*zip(*self.memory))

    def process_return(self, samples):
        obs = torch.Tensor(samples.obs).to(self.device)
        actions = torch.Tensor(samples.action).to(self.device)
        rewards = torch.Tensor(samples.reward).to(self.device)
        done = torch.Tensor(samples.done).to(self.device)
        values = torch.Tensor(samples.value).to(self.device)
        logproba = torch.Tensor(samples.logproba).to(self.device)
        returns = torch.Tensor(len(samples.obs)).to(self.device)
        advantages = torch.Tensor(len(samples.obs)).to(self.device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(len(samples.obs))):
            mask = (1 - done[i])
            returns[i] = rewards[i] + self.gamma * prev_return * mask
            # implement GAE-Lambda advantage calculation
            deltas = rewards[i] + self.gamma * prev_value * mask - values[i]
            advantages[i] = deltas + self.gamma * self.lamda * prev_advantage * mask

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        if self.state_norm:
            obs = (obs - obs.mean(dim=0)) / (obs.std(dim=0) + EPS)
        if self.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        return dict(obs=obs,
                    act=actions,
                    ret=returns,
                    adv=advantages,
                    logproba=logproba, )

    def print_loss(self):
        print('actor_loss', self.record_policy_loss,
              'critic_loss', self.record_vf_loss,
              'entropy', self.record_entropy_loss)
        return self.record_policy_loss, self.record_vf_loss, self.record_entropy_loss
