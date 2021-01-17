import numpy as np
import torch
import random
import torch.optim as optim
from common.networks import *
from common.utils import *
from collections import namedtuple, deque

EPS = 1e-10


class Agent():
    """
    An implementation of the Proximal Policy Optimization (PPO) (by clipping) agent,
    with early stopping based on approximate KL.
    """

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

                 # cppo
                 cost_gamma=None,  # if None, defaults to discount.
                 cost_lambda=None,
                 cost_value_loss_coeff=None,
                 ep_cost_ema_alpha=0,  # 0 for hard update, 1 for no update.
                 objective_penalized=True,  # False for reward-only learning
                 learn_c_value=True,  # Also False for reward-only learning
                 penalty_init=1.,
                 cost_limit=1e-6,
                 cost_scale=1.,  # divides; applied to raw cost and cost_limit
                 normalize_cost_advantage=False,
                 pid_Kp=0.5,
                 pid_Ki=0.1,
                 pid_Kd=1,
                 pid_d_delay=10,
                 pid_delta_p_ema_alpha=0.95,  # 0 for hard update, 1 for no update
                 pid_delta_d_ema_alpha=0.95,
                 sum_norm=True,  # L = (J_r - lam * J_c) / (1 + lam); lam <= 0
                 diff_norm=False,  # L = (1 - lam) * J_r - lam * J_c; 0 <= lam <= 1
                 penalty_max=100,  # only used if sum_norm=diff_norm=False
                 step_cost_limit_steps=None,  # Change the cost limit partway through
                 loss_coeff_cost_value=0.5,  # New value.
                 use_beta_kl=False,
                 use_beta_grad=False,
                 record_beta_kl=False,
                 record_beta_grad=False,
                 clip_grad_norm=1,
                 beta_max=10,
                 beta_ema_alpha=0.9,
                 beta_kl_epochs=1,
                 reward_scale=1,  # multiplicative (unlike cost_scale)
                 lagrange_quadratic_penalty=False,
                 quadratic_penalty_coeff=1,

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
        self.cost_gamma = self.config.agent['cost_gamma'] if 'cost_gamma' in self.config.agent.keys() else cost_gamma
        self.cost_lambda = self.config.agent[
            'cost_lambda'] if 'cost_lambda' in self.config.agent.keys() else cost_lambda
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
        self.clip_grad_norm = clip_grad_norm

        self.pid_Kp = self.config.agent['kp'] if 'kp' in self.config.agent.keys() else pid_Kp
        self.pid_Ki = self.config.agent['ki'] if 'ki' in self.config.agent.keys() else pid_Ki
        self.pid_Kd = self.config.agent['kd'] if 'kd' in self.config.agent.keys() else pid_Kd
        self.cost_limit = self.config.agent['cost_limit'] if 'cost_limit' in self.config.agent.keys() else cost_limit
        self.loss_coeff_cost_value = loss_coeff_cost_value
        self.pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self.pid_delta_d_ema_alpha = pid_delta_d_ema_alpha

        # Main network
        self.policy = CPPOModel(self.obs_dim, self.act_dim, use_layer_norm=layer_norm).to(self.device)
        # Create optimizers
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # PID Variables
        self.pid_i = self.cost_penalty = self.penalty_init = self.config.agent[
            'penalty_init'] if 'penalty_init' in self.config.agent.keys() else penalty_init
        self._delta_p = 0
        self._cost_d = 0
        self.cost_d_pre = 0

        self.record_policy_loss = 0
        self.record_vf_r_loss = 0
        self.record_vf_c_loss = 0
        self.record_entropy_loss = 0

    def set_policy(self, policy):
        self.policy = policy

    def train_model(self, samples, iter_index):
        if not self.config.sampler_gpu_index == self.config.agent_gpu_index:
            self.policy.to(self.device)
        batch_size = len(samples.obs)
        samples, ep_cost_avg = self.process_return(samples)

        self.record_policy_loss = 0
        self.record_vf_r_loss = 0
        self.record_vf_c_loss = 0
        self.record_entropy_loss = 0

        # PID update here:
        delta = float(ep_cost_avg - self.cost_limit)  # ep_cost_avg: tensor
        self.pid_i = max(0., self.pid_i + delta * self.pid_Ki)
        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)

        pid_d = max(0., self._cost_d - self.cost_d_pre)
        pid_o = (self.pid_Kp * self._delta_p + self.pid_i +
                 self.pid_Kd * pid_d)
        self.cost_penalty = max(0., pid_o)
        self.cost_d_pre = self._cost_d

        # ppo update policy
        sample_batch = range(batch_size)
        update_times = int(self.num_epoch * batch_size / self.minibatch_size)
        for i_epoch in range(update_times):
            # choice方法会改变随机种子
            # batch_index = np.random.choice(batch_size, self.minibatch_size, replace=False)
            batch_index = random.sample(sample_batch, self.minibatch_size)
            obs = samples['obs'][batch_index]
            act = samples['act'][batch_index]
            ret = samples['ret'][batch_index]
            adv = samples['adv'][batch_index]
            c_ret = samples['c_ret'][batch_index]
            c_adv = samples['c_adv'][batch_index]
            oldlogproba = samples['logproba'][batch_index]
            action_mean, action_logstd = self.policy.forward_actor(obs)
            newlogproba = normal_logproba(act, action_mean, action_logstd)
            new_r_value = self.policy.forward_r_critic(obs).flatten()
            new_c_value = self.policy.forward_c_critic(obs).flatten()

            ratio = torch.exp(newlogproba - oldlogproba)
            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip) * adv
            pi_loss = - torch.mean(torch.min(surr1, surr2))

            c_surr_1 = ratio * c_adv
            c_surr_2 = ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip) * c_adv
            c_loss = self.cost_penalty * torch.mean(torch.max(c_surr_1, c_surr_2))
            # sum_norm
            pi_loss += c_loss
            pi_loss /= (1 + self.cost_penalty)
            self.record_policy_loss += pi_loss.item()

            loss_value_r = torch.mean((new_r_value - ret).pow(2))
            loss_value_c = torch.mean((new_c_value - c_ret).pow(2))
            self.record_vf_r_loss += loss_value_r.item()
            self.record_vf_c_loss += loss_value_c.item()

            if self.lossvalue_norm:
                loss_value_r = loss_value_r / (ret.std()+1)
                loss_value_c = loss_value_c / (c_ret.std()+1)

            loss_entropy = torch.mean(torch.exp(newlogproba) * newlogproba)
            self.record_entropy_loss += loss_entropy.item()
            total_loss = pi_loss + \
                         self.loss_coeff_value * loss_value_r + \
                         self.loss_coeff_value * loss_value_c + \
                         self.loss_coeff_entropy * loss_entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

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
        self.record_vf_r_loss = self.record_vf_r_loss / update_times
        self.record_vf_c_loss = self.record_vf_c_loss / update_times
        self.record_entropy_loss = self.record_entropy_loss / update_times
    # samples: CPPO_Transition(*zip(*self.memory))
    def process_return(self, samples):
        obs = torch.Tensor(samples.obs).to(self.device)
        actions = torch.Tensor(samples.action).to(self.device)
        rewards = torch.Tensor(samples.reward).to(self.device)
        costs = torch.Tensor(samples.cost).to(self.device)
        done = torch.Tensor(samples.done).to(self.device)
        values = torch.Tensor(samples.value).to(self.device)
        c_values = torch.Tensor(samples.value).to(self.device)
        logproba = torch.Tensor(samples.logproba).to(self.device)
        # sum_cost = torch.Tensor(samples.sum_cost).to(self.device)
        returns = torch.Tensor(len(samples.obs)).to(self.device)
        c_returns = torch.Tensor(len(samples.obs)).to(self.device)
        advantages = torch.Tensor(len(samples.obs)).to(self.device)
        c_advantages = torch.Tensor(len(samples.obs)).to(self.device)

        prev_return = 0
        prev_c_return = 0
        prev_value = 0
        prev_c_value = 0
        prev_advantage = 0
        prev_c_advantage = 0
        for i in reversed(range(len(samples.obs))):
            mask = (1 - done[i])
            returns[i] = rewards[i] + self.gamma * prev_return * mask
            c_returns[i] = costs[i] + self.cost_gamma * prev_c_return * mask
            # implement GAE-Lambda advantage calculation
            deltas = rewards[i] + self.gamma * prev_value * mask - values[i]
            c_deltas = costs[i] + self.cost_gamma * prev_c_value * mask - c_values[i]
            advantages[i] = deltas + self.gamma * self.lamda * prev_advantage * mask
            c_advantages[i] = c_deltas + self.cost_gamma * self.cost_lambda * prev_c_advantage * mask

            prev_return = returns[i]
            prev_c_return = c_returns[i]
            prev_value = values[i]
            prev_c_value = c_values[i]
            prev_advantage = advantages[i]
            prev_c_advantage = c_advantages[i]

        if self.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
            c_advantages = (c_advantages - c_advantages.mean()) / (c_advantages.std() + EPS)

        # ep_cost_avg  = costs.sum() / self.config.episode_size
        # ep_cost_avg = costs.mean()
        ep_cost_avg = c_returns.mean()
        return dict(obs=obs,
                    act=actions,
                    ret=returns,
                    c_ret=c_returns,
                    adv=advantages,
                    c_adv=c_advantages,
                    logproba=logproba,
                    ), ep_cost_avg

    def print_loss(self):
        print('actor_loss', self.record_policy_loss,
              'critic_r_loss', self.record_vf_r_loss,
              'critic_c_loss', self.record_vf_c_loss,
              'entropy', self.record_entropy_loss,
              'alpha', self.cost_penalty)
        return self.record_policy_loss, self.record_vf_r_loss, self.record_vf_c_loss, self.record_entropy_loss, self.cost_penalty
