import torch
import torch.nn as nn
from common.utils import *
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from scipy.stats import norm
from common.utils import *


def identity(x):
    """Return input without any change."""
    return x


class MlpModel(nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,
            use_layer_norm=True  # Module, not Functional.
    ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
                         zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            if use_layer_norm:
                layer_norm(layer=layer, std=1.0)
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            layer = torch.nn.Linear(last_size, output_size)
            if use_layer_norm:
                layer_norm(layer=layer, std=0.01)
            sequence.append(layer)
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
                             else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size


class DDPGActor(MlpModel):
    def __init__(self, input_size,
                 output_size,
                 hidden_sizes=(64, 64),
                 activation=torch.nn.ReLU,
                 output_activation=identity,
                 ):
        super(DDPGActor, self).__init__(input_size, list(hidden_sizes) or [64, 64], output_size, activation, True)

        self.output_activation = output_activation

    def forward(self, x):
        return self.output_activation(super(DDPGActor, self).forward(x))

    def select_action(self, obs, act_noise, iter_index, expl_noise_stop):
        if type(obs) is np.ndarray:
            obs = torch.Tensor(obs)
        action = self.forward(obs)
        action_mean = action.detach().cpu().numpy()
        if iter_index <= expl_noise_stop:
            action = action_mean + ((expl_noise_stop - iter_index) / expl_noise_stop) * \
                     act_noise * np.random.randn(self.output_size)
        else:
            action = action_mean
        return action_mean, action

    def select_action_no_explore(self, obs):
        if type(obs) is np.ndarray:
            obs = torch.Tensor(obs)
        action = self.forward(obs)
        return action


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        # use a simple network for actor. Deeper network does not mean better performance in RL.
        self.net__mid = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )
        lay_dim = mid_dim

        self.net__mean = nn.Linear(lay_dim, action_dim)
        self.net__std_log = nn.Linear(lay_dim, action_dim)

        layer_norm(self.net__mean, std=0.01)  # net[-1] is output layer for action, it is no necessary.

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def forward(self, state, noise_std=0.0):  # in fact, noise_std is a boolean
        x = self.net__mid(state)
        a_mean = self.net__mean(x)  # NOTICE! it is a_mean without .tanh()

        if noise_std != 0.0:
            a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a_mean = torch.normal(a_mean, a_std)  # NOTICE! it needs .tanh()
        return a_mean.tanh()

    def get_a_log_prob(self, state):
        x = self.net__mid(state)
        a_mean = self.net__mean(x)  # NOTICE! it needs a_mean.tanh()
        a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        # self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi

        # same as below:
        # from torch.distributions.normal import Normal
        # log_prob_noise = Normal(a_mean, a_std).log_prob(a_noise)
        # same as below:
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))

        a_noise_tanh = a_noise.tanh()
        log_prob = -log_prob_noise - (-a_noise_tanh.pow(2) + 1.000001).log()

        # same as below:
        # epsilon = 1e-6
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)

    def select_action(self, state, noise_std=0.0):
        if type(state) is np.ndarray:
            state = torch.Tensor(state)
        action = self.forward(state, noise_std=noise_std).detach().cpu().numpy()
        return action


class SACActor(torch.nn.Module):
    def __init__(
            self,
            input_size, output_size,
            output_limit=1.0,
            hidden_sizes=[64, 64],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            use_layer_norm=True,
            action_modifier=None
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.actor_output_nonlinearity = torch.nn.Tanh,
        # self.critic_output_nonlinearity = torch.nn.Tanh,
        self.device = device
        self.action_modifier = action_modifier
        self.actor = torch.nn.Sequential(nn.Linear(input_size, self.hidden_sizes[0]), nn.ReLU(),
                                         nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]), nn.ReLU(), )

        self.actor_mean = nn.Linear(self.hidden_sizes[-1], output_size)
        self.actor_logstd = nn.Linear(self.hidden_sizes[-1], output_size)

        if use_layer_norm:
            layer_norm(self.actor, std=0.01)
            layer_norm(self.actor_mean, std=0.01)
            layer_norm(self.actor_logstd, std=0.01)

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = math.log(math.sqrt(2 * math.pi))

    def forward(self, state, noise_std=0.0):
        x = self.actor(state)
        a_mean = self.actor_mean(x)  # NOTICE! it is a_mean without .tanh()
        if noise_std != 0.0:
            a_std_log = self.actor_logstd(x).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a_mean = torch.normal(a_mean, a_std)  # NOTICE! it needs .tanh()
        return a_mean.tanh()

    def get_a_log_prob(self, state):  # actor
        x = self.actor(state)
        a_mean = self.actor_mean(x)
        a_log_std = self.actor_logstd(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)
        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        log_prob_noise = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)

        a_noise_tanh = a_noise.tanh()
        correction = -(-a_noise_tanh.pow(2) + 1.000001).log()
        # log(1 - tanh(x)^2)= 2 * (log(2) - x - softplus(-2x))
        # correction = - 2. * (math.log(2.) - a_noise - torch.nn.functional.softplus(-2. * a_noise))
        log_prob = log_prob_noise + correction
        return a_noise_tanh, log_prob.sum(1, keepdim=True)

    def compute_log_prob(self, state, a_noise):
        x = self.actor(state)
        a_mean = self.actor_mean(x)
        a_log_std = self.actor_logstd(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return log_prob.sum(1)

    def set_action_modifier(self, action_modifier):
        self.action_modifier = action_modifier

    def select_action(self, state, noise_std=0.0, c=[0]):
        if type(state) is np.ndarray:
            state = torch.Tensor(state)
        action = self.forward(state, noise_std=noise_std).detach().cpu().numpy()
        if self.action_modifier == None:
            return action
        else:
            return self.action_modifier(state, action, c)

    def test(self, state):
        x = self.actor(state)
        a_mean = self.actor_mean(x)
        a_std_log = self.actor_logstd(x).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()
        return a_mean, a_std


class DDPGCritic(MlpModel):
    def __init__(self, input_size,
                 output_size,
                 hidden_sizes=(64, 64),
                 activation=torch.nn.ReLU,
                 output_activation=identity,
                 ):
        super(DDPGCritic, self).__init__(input_size, list(hidden_sizes) or [64, 64], output_size, activation, True)
        self.output_activation = output_activation

    def forward(self, x, a):
        q = torch.cat([x, a], dim=-1)
        q = super(DDPGCritic, self).forward(q)
        return self.output_activation(q)


class CriticTwin(nn.Module):  # TwinSAC <- TD3(TwinDDD) <- DoubleDQN <- Double Q-learning
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def build_critic_network():
            net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, 1), )
            layer_norm(net[-1], std=0.01)  # It is no necessary.
            return net

        self.net1 = build_critic_network()
        self.net2 = build_critic_network()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value = self.net1(x)
        return q_value

    def get_q1_q2(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value1 = self.net1(x)
        q_value2 = self.net2(x)
        return q_value1, q_value2


class DoubleQCritic(torch.nn.Module):
    def __init__(self, input_size,
                 output_size,
                 hidden_sizes=(64, 64),
                 activation=torch.nn.ReLU,
                 output_activation=identity,
                 ):
        super().__init__()
        self.Q1 = MlpModel(input_size, list(hidden_sizes), output_size, activation, True)
        self.Q2 = MlpModel(input_size, list(hidden_sizes), output_size, activation, True)
        self.output_activation = output_activation

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value = self.Q1(x)
        return q_value

    # def forward(self, s, a):
    #     sa = torch.cat([s, a], dim=-1)
    #     q1 = self.Q1(sa)
    #     q2 = self.Q2(sa)
    #     return self.output_activation(q1), self.output_activation(q2)

    def get_q1_q2(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.Q1(sa)
        q2 = self.Q2(sa)
        return q1, q2
    #
    # def Q1_forward(self, s, a):
    #     sa = torch.cat([s, a], dim=-1)
    #     return self.output_activation(self.Q1.forward(sa))


class SACModel(torch.nn.Module):
    def __init__(
            self,
            input_size, output_size,
            output_limit=1.0,
            hidden_sizes=[64, 64],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            use_layer_norm=True
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.actor_output_nonlinearity = torch.nn.Tanh,
        # self.critic_output_nonlinearity = torch.nn.Tanh,
        self.device = device

        self.actor = torch.nn.Sequential(nn.Linear(input_size, self.hidden_sizes[0]), nn.ReLU(),
                                         nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]), nn.ReLU())

        self.actor_mean = nn.Linear(self.hidden_sizes[-1], output_size)
        self.actor_logstd = nn.Linear(self.hidden_sizes[-1], output_size)

        self.Q1 = torch.nn.Sequential(nn.Linear(input_size + output_size, self.hidden_sizes[0]), nn.ReLU(),
                                      nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]), nn.ReLU(),
                                      nn.Linear(self.hidden_sizes[1], 1))

        self.Q2 = torch.nn.Sequential(nn.Linear(input_size + output_size, self.hidden_sizes[0]), nn.ReLU(),
                                      nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]), nn.ReLU(),
                                      nn.Linear(self.hidden_sizes[1], 1))

        if use_layer_norm:
            layer_norm(self.actor, std=0.01)
            layer_norm(self.Q1, std=0.01)
            layer_norm(self.Q2, std=0.01)
            layer_norm(self.actor_mean, std=0.01)
            layer_norm(self.actor_logstd, std=0.01)

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = 0.5 * math.log(math.sqrt(2 * math.pi))

    def forward(self, state, noise_std=0.0):
        x = self.actor(state)
        a_mean = self.actor_mean(x)  # NOTICE! it is a_mean without .tanh()
        if noise_std != 0.0:
            a_std_log = self.actor_logstd(x).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a = torch.normal(a_mean, a_std)  # NOTICE! it needs .tanh()
        return a.tanh()

    def get_q1_q2(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.Q1(sa)
        q2 = self.Q2(sa)
        return q1, q2

    def get_a_log_prob(self, state):  # actor
        x = self.actor(state)
        a_mean = self.actor_mean(x)
        a_log_std = self.actor_logstd(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)
        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True)
        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return a_mean, a_log_std, a_noise, log_prob.sum(1)

    def compute_log_prob(self, state, a_noise):
        x = self.actor(state)
        a_mean = self.actor_mean(x)
        a_log_std = self.actor_logstd(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return log_prob.sum(1)

    def select_action(self, state, noise_std=0.0):
        if type(state) is np.ndarray:
            state = torch.Tensor(state)
        action = self.forward(state, noise_std=noise_std).detach().cpu().numpy()
        return action


class PPOModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_layer_norm=True):
        super().__init__()
        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if use_layer_norm:
            layer_norm(self.actor_fc1, std=1.0)
            layer_norm(self.actor_fc2, std=1.0)
            layer_norm(self.actor_fc3, std=0.01)

            layer_norm(self.critic_fc1, std=1.0)
            layer_norm(self.critic_fc2, std=1.0)
            layer_norm(self.critic_fc3, std=1.0)

    def forward(self, states):
        action_mean, action_logstd = self.forward_actor(states)
        return action_mean

    def forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self.forward_actor(states)
        logproba = normal_logproba(actions, action_mean, action_logstd)
        return logproba

    def select_action(self, obs):
        if type(obs) is np.ndarray:
            obs = torch.Tensor(obs)
        action_mean, action_logstd = self.forward_actor(obs.unsqueeze(0))
        value = self.forward_critic(obs.unsqueeze(0))
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        logproba = normal_logproba(action, action_mean, action_logstd)
        return action, logproba, value, action_mean, action_std


class PPOModel2(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_norm=True):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3_sigma = nn.Linear(64, num_outputs)
        self.actor_fc3_mu = nn.Linear(64, num_outputs)

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3_sigma, std=0.01)
            self.layer_norm(self.actor_fc3_mu, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        action_mean, action_logstd = self.forward_actor(states)
        critic_value = self.forward_critic(states)
        return action_mean, action_logstd, critic_value

    def forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3_sigma(x)
        action_logstd = self.actor_fc3_mu(x)
        return action_mean, action_logstd

    def forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def get_logproba(self, observation, actions):
        action_mean, action_logstd = self.forward_actor(observation)
        return normal_logproba(actions, action_mean, action_logstd)

    def select_action(self, obs):
        if type(obs) is np.ndarray:
            obs = torch.Tensor(obs)
        action_mean, action_logstd, value = self.forward(obs.unsqueeze(0))
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        logproba = normal_logproba(action, action_mean, action_logstd)
        return action, logproba, value, action_mean, action_std


class CPPOModel(nn.Module):

    def __init__(self, num_inputs, num_outputs, use_layer_norm=True):
        super().__init__()
        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_r_fc1 = nn.Linear(num_inputs, 64)
        self.critic_r_fc2 = nn.Linear(64, 64)
        self.critic_r_fc3 = nn.Linear(64, 1)

        self.critic_c_fc1 = nn.Linear(num_inputs, 64)
        self.critic_c_fc2 = nn.Linear(64, 64)
        self.critic_c_fc3 = nn.Linear(64, 1)

        if use_layer_norm:
            layer_norm(self.actor_fc1, std=1.0)
            layer_norm(self.actor_fc2, std=1.0)
            layer_norm(self.actor_fc3, std=0.01)

            layer_norm(self.critic_r_fc1, std=1.0)
            layer_norm(self.critic_r_fc2, std=1.0)
            layer_norm(self.critic_r_fc3, std=1.0)

            layer_norm(self.critic_c_fc1, std=1.0)
            layer_norm(self.critic_c_fc2, std=1.0)
            layer_norm(self.critic_c_fc3, std=1.0)

    def forward(self, states):
        action_mean, action_logstd = self.forward_actor(states)
        return action_mean

    def forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def forward_r_critic(self, obs):
        x = torch.tanh(self.critic_r_fc1(obs))
        x = torch.tanh(self.critic_r_fc2(x))
        critic_value = self.critic_r_fc3(x)
        return critic_value

    def forward_c_critic(self, obs):
        x = torch.tanh(self.critic_c_fc1(obs))
        x = torch.tanh(self.critic_c_fc2(x))
        critic_value = self.critic_c_fc3(x)
        return critic_value

    def get_logproba(self, obs, actions):
        action_mean, action_logstd = self.forward_actor(obs)
        logproba = normal_logproba(actions, action_mean, action_logstd)
        return logproba

    def select_action(self, obs):
        if type(obs) is np.ndarray:
            obs = torch.Tensor(obs)
        obs = obs.unsqueeze(0)
        action_mean, action_logstd = self.forward_actor(obs)
        r_value, c_value = self.forward_r_critic(obs), self.forward_c_critic(obs)
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        logproba = normal_logproba(action, action_mean, action_logstd)
        return action, logproba, r_value, c_value, action_mean, action_std


# class CPPOModel(torch.nn.Module):
#     def __init__(
#             self,
#             input_size, output_size,
#             output_limit=1.0,
#             hidden_sizes=(64, 64),
#             hidden_nonlinearity=torch.nn.Tanh,
#             mu_nonlinearity=None,  # mu_nonlinearity=torch.nn.Tanh,
#             init_log_std=0.,
#             device=torch.device('cpu'),
#             use_layer_norm=True
#     ):
#         super().__init__()
#
#         self.input_size = input_size
#         self.output_size = output_size
#         self.output_limit = output_limit
#         self.hidden_sizes = hidden_sizes
#         self.hidden_nonlinearity = hidden_nonlinearity
#         self.mu_nonlinearity = mu_nonlinearity
#         self.device = device
#
#         policy = MlpModel(
#             input_size=input_size,
#             hidden_sizes=list(hidden_sizes) or [64, 64],
#             output_size=output_size,
#             nonlinearity=self.hidden_nonlinearity,
#             use_layer_norm=use_layer_norm
#         )
#         if self.mu_nonlinearity is not None:
#             self.policy = torch.nn.Sequential(policy, self.mu_nonlinearity()).to(self.device)
#         else:
#             self.policy = policy
#
#         self.r_critic = MlpModel(
#             self.input_size,
#             list(hidden_sizes) or [64, 64],
#             1,
#             nonlinearity=self.hidden_nonlinearity,
#             use_layer_norm=use_layer_norm).to(self.device)
#
#         self.c_critic = MlpModel(
#             self.input_size,
#             list(hidden_sizes) or [64, 64],
#             1,
#             nonlinearity=self.hidden_nonlinearity,
#             use_layer_norm=use_layer_norm).to(self.device)
#
#         self.policy_logstd = torch.nn.Parameter(init_log_std * torch.ones(self.output_size))
#
#     def forward(self, observation):
#         action_mean = self.policy(observation)
#         return action_mean
#
#     def forward_policy(self, obs):
#         action_mean = self.policy(obs)
#         action_logstd = self.policy_logstd.expand_as(action_mean)
#         return action_mean, action_logstd
#
#     def forward_r_critic(self, obs):
#         return self.r_critic(obs)
#
#     def forward_c_critic(self, obs):
#         return self.c_critic(obs)
#
#     def get_logproba(self, observation, actions):
#         """
#         return probability of chosen the given actions under corresponding states of current network
#         :param states: Tensor
#         :param actions: Tensor
#         """
#         action_mean = self.policy(observation)
#         action_logstd = self.policy_logstd.expand_as(action_mean)
#         logproba = normal_logproba(actions, action_mean, action_logstd)
#         return logproba
#
#     def select_action(self, obs):
#         if type(obs) is np.ndarray:
#             obs = torch.Tensor(obs).to(self.device)
#         action_mean, action_logstd = self.forward_policy(obs.unsqueeze(0))
#         r_value, c_value = self.forward_r_critic(obs.unsqueeze(0)), self.forward_c_critic(obs.unsqueeze(0))
#         action_std = torch.exp(action_logstd)
#         action = torch.normal(action_mean, action_std)
#         logproba = normal_logproba(action, action_mean, action_logstd)
#         return action, logproba, r_value, c_value, action_mean, action_std


# cost netwaork 和 value network 融合
class CPPOModel2(torch.nn.Module):
    def __init__(
            self,
            input_size, output_size,
            output_limit=1.0,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=torch.nn.Tanh,
            mu_nonlinearity=torch.nn.Tanh,
            init_log_std=0.,
            device=torch.device('cpu'),
            use_layer_norm=True
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.mu_nonlinearity = mu_nonlinearity
        self.device = device

        policy = MlpModel(
            input_size=input_size,
            hidden_sizes=list(hidden_sizes) or [64, 64],
            output_size=output_size,
            nonlinearity=self.hidden_nonlinearity,
            use_layer_norm=use_layer_norm
        )
        if self.mu_nonlinearity is not None:
            self.policy = torch.nn.Sequential(policy, self.mu_nonlinearity()).to(self.device)
        else:
            self.policy = policy

        self.critic = MlpModel(
            self.input_size,
            list(hidden_sizes) or [64, 64],
            1,
            nonlinearity=self.hidden_nonlinearity,
            use_layer_norm=use_layer_norm).to(self.device)

        self.policy_logstd = torch.nn.Parameter(init_log_std * torch.ones(self.output_size))

    def forward(self, observation):
        action_mean = self.policy(observation)
        return action_mean

    def forward_policy(self, obs):
        action_mean = self.policy(obs)
        action_logstd = self.policy_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def forward_critic(self, obs):
        return self.critic(obs)

    def get_logproba(self, observation, actions):

        action_mean = self.policy(observation)
        action_logstd = self.policy_logstd.expand_as(action_mean)
        logproba = normal_logproba(actions, action_mean, action_logstd)
        return logproba

    def select_action(self, obs):
        if type(obs) is np.ndarray:
            obs = torch.Tensor(obs).to(self.device)
        action_mean, action_logstd = self.forward_policy(obs.unsqueeze(0))
        value = self.forward_critic(obs.unsqueeze(0))
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        logproba = normal_logproba(action, action_mean, action_logstd)
        return action, logproba, value, -value, action_mean, action_std


class CPPOModel3(torch.nn.Module):
    def __init__(
            self,
            input_size, output_size,
            output_limit=1.0,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=torch.nn.Tanh,
            mu_nonlinearity=torch.nn.Tanh,
            init_log_std=0.,
            device=torch.device('cpu'),
            layer_norm=True
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.mu_nonlinearity = mu_nonlinearity
        self.device = device

        self.body = MlpModel(
            input_size=input_size,
            hidden_sizes=list(hidden_sizes) or [64, 64],
            nonlinearity=self.hidden_nonlinearity,
            layer_norm=layer_norm
        )
        last_size = self.body.output_size
        mu_linear = MlpModel(input_size=last_size,
                             hidden_sizes=None,
                             output_size=self.output_size,
                             layer_norm=layer_norm)
        if self.mu_nonlinearity is not None:
            self.policy = torch.nn.Sequential(mu_linear, self.mu_nonlinearity()).to(self.device)
        else:
            self.policy = mu_linear

        self.r_critic = MlpModel(input_size=last_size,
                                 hidden_sizes=None,
                                 output_size=self.output_size,
                                 layer_norm=layer_norm)
        self.c_critic = MlpModel(input_size=last_size,
                                 hidden_sizes=None,
                                 output_size=self.output_size,
                                 layer_norm=layer_norm)

        self.policy_logstd = torch.nn.Parameter(init_log_std * torch.ones(self.output_size))

    def forward(self, obs):
        feature = self.body(obs)
        action_mean = self.policy(feature)
        action_logstd = self.policy_logstd.expand_as(action_mean)
        r_value = self.r_critic(feature)
        c_value = self.c_critic(feature)
        return action_mean, action_logstd, r_value, c_value

    def forward_policy(self, obs):
        feature = self.body(obs)
        action_mean = self.policy(feature)
        action_logstd = self.policy_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def forward_r_critic(self, obs):
        feature = self.body(obs)
        return self.r_critic(feature)

    def forward_c_critic(self, obs):
        feature = self.body(obs)
        return self.c_critic(feature)

    def get_logproba(self, obs, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, _ = self.forward_policy(obs)
        action_logstd = self.policy_logstd.expand_as(action_mean)
        logproba = normal_logproba(actions, action_mean, action_logstd)
        return logproba

    def select_action(self, obs):
        if type(obs) is np.ndarray:
            obs = torch.Tensor(obs).to(self.device)
        action_mean, action_logstd, r_value, c_value = self.forward(obs.unsqueeze(0))
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        logproba = normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba, r_value, c_value, action_mean, action_std

#
# class CPPOModel_test(nn.Module):
#     def __init__(self, num_inputs, num_outputs, layer_norm=True):
#         super().__init__()
#         self.actor_fc1 = nn.Linear(num_inputs, 64)
#         self.actor_fc2 = nn.Linear(64, 64)
#         self.actor_fc3_sigma = nn.Linear(64, num_outputs)
#         self.actor_fc3_mu = nn.Linear(64, num_outputs)
#
#         self.critic_r_fc1 = nn.Linear(num_inputs, 64)
#         self.critic_r_fc2 = nn.Linear(64, 64)
#         self.critic_r_fc3 = nn.Linear(64, 1)
#
#         self.critic_c_fc1 = nn.Linear(num_inputs, 64)
#         self.critic_c_fc2 = nn.Linear(64, 64)
#         self.critic_c_fc3 = nn.Linear(64, 1)
#
#         if layer_norm:
#             self.layer_norm(self.actor_fc1, std=1.0)
#             self.layer_norm(self.actor_fc2, std=1.0)
#             self.layer_norm(self.actor_fc3_sigma, std=0.01)
#             self.layer_norm(self.actor_fc3_mu, std=0.01)
#
#             self.layer_norm(self.critic_r_fc1, std=1.0)
#             self.layer_norm(self.critic_r_fc2, std=1.0)
#             self.layer_norm(self.critic_r_fc3, std=1.0)
#
#             self.layer_norm(self.critic_c_fc1, std=1.0)
#             self.layer_norm(self.critic_c_fc2, std=1.0)
#             self.layer_norm(self.critic_c_fc3, std=1.0)
#
#     @staticmethod
#     def layer_norm(layer, std=1.0, bias_const=0.0):
#         torch.nn.init.orthogonal_(layer.weight, std)
#         torch.nn.init.constant_(layer.bias, bias_const)
#
#     def forward(self, states):
#         """
#         run policy network (actor) as well as value network (critic)
#         :param states: a Tensor2 represents states
#         :return: 3 Tensor2
#         """
#         action_mean, action_logstd = self.forward_policy(states)
#         r_value = self.forward_r_critic(states)
#         c_value = self.forward_c_critic(states)
#         return action_mean, action_logstd, r_value, c_value
#
#     def forward_policy(self, states):
#         x = torch.tanh(self.actor_fc1(states))
#         x = torch.tanh(self.actor_fc2(x))
#         action_mean, action_logstd = self.actor_fc3_sigma(x), self.actor_fc3_mu(x)
#         return action_mean, action_logstd
#
#     def forward_r_critic(self, states):
#         x = torch.tanh(self.critic_r_fc1(states))
#         x = torch.tanh(self.critic_r_fc2(x))
#         critic_value = self.critic_r_fc3(x)
#         return critic_value
#
#     def forward_c_critic(self, states):
#         x = torch.tanh(self.critic_c_fc1(states))
#         x = torch.tanh(self.critic_c_fc2(x))
#         critic_value = self.critic_c_fc3(x)
#         return critic_value
#
#     def get_logproba(self, states, actions):
#         """
#         return probability of chosen the given actions under corresponding states of current network
#         :param states: Tensor
#         :param actions: Tensor
#         """
#         action_mean, action_logstd = self.forward_policy(states)
#         logproba = normal_logproba(actions, action_mean, action_logstd)
#         return logproba
#
#     def select_action(self, obs):
#         if type(obs) is np.ndarray:
#             obs = torch.Tensor(obs)
#         action_mean, action_logstd, r_value, c_value = self.forward(obs.unsqueeze(0))
#         action_std = torch.exp(action_logstd)
#         action = torch.normal(action_mean, action_std)
#         logproba = normal_logproba(action, action_mean, action_logstd)
#         return action, logproba, r_value, c_value, action_mean, action_std
