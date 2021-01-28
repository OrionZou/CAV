"""Thanks for ZenYiYan, GitHub: YonV1943 ElegantRL
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from common.utils import *
from common.buffers import *
from common.networks import *

EPS = 1e-10


class Agent(object):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 config,
                 automatic_entropy_tuning=True,
                 hidden_sizes=(256, 256),
                 action_modifier=None,
                 lr=3e-4,
                 minibatch_size=256,
                 num_epoch=1,
                 ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        if config.gpu_index == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device(
                'cpu')

        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.hidden_sizes = hidden_sizes
        self.reward_scale = self.config.agent['reward_scale']
        self.lr = self.config.agent['lr'] if 'lr' in self.config.agent.keys() else lr
        self.minibatch_size = self.config.agent[
            'minibatch_size'] if 'minibatch_size' in self.config.agent.keys() else minibatch_size
        self.num_epoch = self.config.agent['num_epoch'] if 'num_epoch' in self.config.agent.keys() else num_epoch

        # Main network
        # self.policy = ActorSAC(self.obs_dim ,self.act_dim, self.hidden_sizes[-1]).to(self.device)
        self.policy = SACActor(self.obs_dim, self.act_dim,
                               hidden_sizes=self.hidden_sizes,
                               device=self.device,action_modifier=action_modifier).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.qf = DoubleQCritic(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
        # critic_dim = int(self.hidden_sizes[-1] * 1.25)
        # self.qf = CriticTwin(self.obs_dim ,self.act_dim, critic_dim).to(self.device)
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=self.lr)

        # Initialize target parameters to match main parameters
        self.policy_traget = SACActor(self.obs_dim, self.act_dim,
                                      hidden_sizes=self.hidden_sizes,
                                      device=self.device).to(self.device)
        self.policy_traget.eval()
        self.policy_traget.load_state_dict(self.policy_traget.state_dict())

        self.qf_target = DoubleQCritic(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
        # self.qf_target = CriticTwin(self.obs_dim, self.act_dim, critic_dim).to(self.device)
        self.qf_target.eval()
        self.qf_target.load_state_dict(self.qf.state_dict())

        # self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.MSELoss()
        # If automatic entropy tuning is True,
        # initialize a target entropy, a log alpha and an alpha optimizer
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.lr)
            self.target_entropy = math.log(act_dim + 1) * 0.5
            # self.target_entropy=math.log(act_dim) * 0.98

        self.record_policy_loss = 0
        self.record_vf_loss = 0
        self.record_alpha_loss = 0

    def train_model(self, buffer,iter_index):
        self.policy.to(self.device)

        self.record_policy_loss = 0
        self.record_vf_loss = 0
        self.record_alpha_loss = 0

        alpha = self.log_alpha.exp()  # auto temperature parameter
        k = 1.0 + buffer.size / buffer.max_size
        minibatch_size = int(self.minibatch_size * k)  # increase batch_size
        update_times = int(self.config.env['max_step'] * k)  # increase training_step
        for i in range(1, update_times * self.num_epoch + 1):
            with torch.no_grad():
                batch_index = np.random.choice(buffer.size, minibatch_size, replace=True)
                obs, actions, rewards, obs_next, mask = self.process_return(buffer.sample(batch_index))
                # next_a_noise, next_log_prob = self.policy_traget.get_a_log_prob(obs)
                next_a_noise, next_log_prob = self.policy.get_a_log_prob(obs_next)
                next_q_target = torch.min(*self.qf_target.get_q1_q2(obs_next, next_a_noise))  # twin critic

                q_target = rewards * self.reward_scale + mask * (next_q_target - next_log_prob * alpha)

            '''critic_loss'''
            q1_value, q2_value = self.qf.get_q1_q2(obs, actions)
            qf_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            self.record_vf_loss += qf_loss.item() * 0.5
            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            '''actor_loss'''
            if i % self.num_epoch == 0:
                '''stochastic policy'''
                a_noise, log_prob = self.policy.get_a_log_prob(obs)  # policy gradient
                '''auto temperature parameter: alpha'''
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.record_alpha_loss += alpha_loss.item()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()

                # q_eval_pg = self.qf(obs, a_noise) # policy gradient,
                q_eval_pg = torch.min(*self.qf.get_q1_q2(obs, a_noise))  # policy gradient, stable but slower
                policy_loss = -(q_eval_pg - log_prob * alpha).mean()  # policy gradient
                self.record_policy_loss += policy_loss.item()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

            """target update"""
            soft_target_update(self.qf, self.qf_target)  # soft target update
            soft_target_update(self.policy_traget, self.policy_traget)
            # if qf_loss > 1e3:
            #     print(log_prob)
            #     print(alpha)
            #     print(qf_loss)
        self.record_policy_loss = self.record_policy_loss / update_times
        self.record_alpha_loss = self.record_alpha_loss / update_times
        self.record_vf_loss = self.record_vf_loss / (update_times * self.num_epoch)

    def process_return(self, samples):
        obs = torch.Tensor(samples.obs).to(self.device)
        actions = torch.Tensor(samples.action).to(self.device)
        rewards = torch.Tensor(samples.reward).to(self.device)
        done = torch.Tensor(samples.done).to(self.device)
        obs_next = torch.Tensor(samples.obs_next).to(self.device)
        mask = (1 - done) * self.config.agent['gamma']
        rewards = torch.unsqueeze(rewards, 1)
        mask = torch.unsqueeze(mask, 1)
        # obs = (obs - obs.mean(dim=0)) / (obs.std(dim=0) + EPS)
        return obs, actions, rewards, obs_next, mask

    def print_loss(self):
        print('actor_loss', self.record_policy_loss,
              'critic_loss', self.record_vf_loss,
              'alpha_loss', self.record_alpha_loss)
        return self.record_policy_loss, self.record_vf_loss, self.record_alpha_loss
