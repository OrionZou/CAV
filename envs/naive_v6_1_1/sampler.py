from torch.multiprocessing import Process, Value
import torch
import torch.multiprocessing as mp
import numpy as np
from collections import namedtuple
from common.buffers import Episode, Memory, ReplayBuffer
from common.tensorboard import TensorBoard, show_train, show_evel
import gym
import time

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.multiprocessing.set_start_method('spawn', force=True)
# torch.multiprocessing.set_sharing_strategy('file_system')

PPO_Transition = namedtuple('PPO_Transition', ('obs', 'action', 'reward', 'done', 'value', 'logproba'))

CPPO_Transition = namedtuple('CPPO_Transition',
                             ('obs', 'action', 'reward', 'cost', 'sum_cost', 'done', 'value', 'c_value', 'logproba'))

TD_Transition = namedtuple('TD_Transition', ('obs', 'action', 'obs_next', 'reward', 'done'))

Safe_TD_Transition = namedtuple('Safe_TD_Transition',
                                ('obs', 'action', 'obs_next', 'reward', 'cost', 'cost_next', 'done'))

# 0正常 1阻塞 主进程完成一次sample后，控制子进程阻塞
Sub_Proc_Blocking = Value('i', 0)


def make_env(config_env, seed, env_index):
    env = gym.make(config_env['id'])
    env.start(str(env_index), path=config_env['path'], gui=config_env['gui'], max_step=config_env['max_step'],
              reward_model=config_env['reward_model'], is_human_model=config_env['human_model'])
    env.seed(seed)
    return env


def is_on_policy(args_algo):
    on_policys = ['ppo2', 'trpo', 'cppo', 'cppo2']
    if args_algo in on_policys:
        return True
    else:
        return False


class EnvWorker(Process):
    def __init__(self, id, remote, send_queue, send_lock, config, Transition=TD_Transition):
        """
        针对单个环境的worker
        :param env_fn: 建立好的一个环境
        :param recv_queue: worker 接受消息队列
        :param send_queue: worker 发送消息队列
        :param recv_lock: 接受消息队列的锁
        :param send_lock: 发送消息队列的锁
        """
        super(EnvWorker, self).__init__()

        self.id = id
        self.remote = remote
        self.send_queue = send_queue
        self.send_lock = send_lock
        self.config = config

        self.episode = Episode(Transition)
        self.result = {}

    def run(self):
        # seed = self.config.seed + self.id
        seed = self.config.seed
        self.env = make_env(config_env=self.config.env, seed=seed, env_index=self.id)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        while True:
            # print('id:', np.random.get_state()[1][0], 'seed:', np.random.get_state()[1][-1])
            # 子进程接受来自主进程的命令
            policy, iter_index = self.remote.recv()
            results = {}
            results['reward'] = 0
            results['time'] = 0
            results['speed'] = 0
            results['sum_cost'] = 0
            results['cost'] = 0
            results['collision_cost'] = 0
            results['collision'] = 0
            results['success'] = 0
            results['fuel'] = 0
            results['AEB_num'] = 0
            results['exploration'] = []
            results['action_std'] = []

            if iter_index == -1:
                self.episode.clear()
                obs = self.env.reset()
                done = False
                episode_length = 0
                cost = 0
                while not done:
                    # 主进程完成一次sample后，正在sample的子进程结束当前sample
                    if Sub_Proc_Blocking.value == 1:
                        break
                    obs = torch.Tensor(obs)
                    action = policy(obs.unsqueeze(0))
                    action = action.detach().cpu().numpy()[0]
                    if self.config.agent['name'] == 'safe_sac':
                        action = policy.action_modifier(obs, action, [cost])
                    obs, reward, done, info = self.env.step(action)
                    cost_next = info['cost']
                    cost = cost_next
                    episode_length += 1
                    results['reward'] += reward
                    results['speed'] += info['speed']
                    results['cost'] += info['cost']
                    results['sum_cost'] += info['sum_cost']
                    results['collision_cost'] += info['collision_cost']
                    results['collision'] += info['crash']
                    results['success'] += info['success']
                    results['fuel'] += info['fuel']
                    results['AEB_num'] += info['AEB_num']
                # 主进程正在sample时，结束sample的子进程向主进程发送消息
                if Sub_Proc_Blocking.value == 0:
                    results['time'] = 0.2 * episode_length
                    results['speed'] = results['speed'] / episode_length
                    self.send_lock.acquire()
                    self.send_queue.put((self.id, results))
                    self.send_lock.release()
                continue

            if self.config.agent['name'] == 'ppo2':
                self.episode.clear()
                obs = self.env.reset()
                done = False

                while not done:
                    a, a_logprob, v, a_mean, a_std = policy.select_action(obs)
                    a = a.detach().cpu().numpy()[0]
                    v = v.detach().cpu().numpy()[0]
                    a_logprob = a_logprob.detach().cpu().numpy()[0]
                    a_std = a_std.detach().cpu().numpy()[0]
                    a_mean = a_mean.detach().cpu().numpy()[0]
                    obs_next, r, done, info = self.env.step(a)
                    self.episode.push(obs, a, r, done, v, a_logprob)
                    obs = obs_next
                    results['reward'] += r
                    results['speed'] += info['speed']
                    results['cost'] += info['cost']
                    results['sum_cost'] += info['sum_cost']
                    results['collision_cost'] += info['collision_cost']
                    if info['crash']:
                        results['collision'] = 1
                    results['success'] += info['success']
                    # results['success'] += info['done']
                    results['fuel'] += info['fuel']
                    results['AEB_num'] += info['AEB_num']
                    results['exploration'].append(np.abs(a.clip(-1, 1) - a_mean.clip(-1, 1)))
                    results['action_std'].append(a_std)
                    # 主进程完成一次sample后，正在sample的子进程结束当前sample
                    if Sub_Proc_Blocking.value == 1:
                        break

                # 主进程正在sample时，结束sample的子进程向主进程发送消息
                if Sub_Proc_Blocking.value == 0:
                    results['time'] = 0.2 * len(self.episode)
                    results['speed'] = results['speed'] / len(self.episode)

                    self.send_lock.acquire()
                    self.send_queue.put((self.id, self.episode, results))
                    self.send_lock.release()
            elif self.config.agent['name'] == 'cppo':
                self.episode.clear()
                obs = self.env.reset()
                done = False

                while not done:
                    a, a_logprob, r_v, c_v, a_mean, a_std = policy.select_action(obs)
                    a = a.detach().cpu().numpy()[0]
                    r_v = r_v.detach().cpu().numpy()[0]
                    c_v = c_v.detach().cpu().numpy()[0]
                    a_logprob = a_logprob.detach().cpu().numpy()[0]
                    a_mean = a_mean.detach().cpu().numpy()[0]
                    a_std = a_std.detach().cpu().numpy()[0]
                    obs_next, r, done, info = self.env.step(a)
                    self.episode.push(obs, a, r, info['cost'], info['sum_cost'], done, r_v, c_v, a_logprob)
                    obs = obs_next
                    results['reward'] += r
                    results['speed'] += info['speed']
                    results['cost'] += info['cost']
                    results['sum_cost'] += info['sum_cost']
                    results['collision_cost'] += info['collision_cost']
                    if info['crash']:
                        results['collision'] = 1
                    results['success'] += info['success']
                    results['fuel'] += info['fuel']
                    results['exploration'].append(np.abs(a.clip(-1, 1) - a_mean.clip(-1, 1)))
                    results['action_std'].append(a_std)
                    results['AEB_num'] += info['AEB_num']
                    # 主进程完成一次sample后，正在sample的子进程结束当前sample
                    if Sub_Proc_Blocking.value == 1:
                        break

                # 主进程正在sample时，结束sample的子进程向主进程发送消息
                if Sub_Proc_Blocking.value == 0:
                    results['time'] = 0.2 * len(self.episode)
                    results['speed'] = results['speed'] / len(self.episode)

                    self.send_lock.acquire()
                    self.send_queue.put((self.id, self.episode, results))
                    self.send_lock.release()

            elif self.config.agent['name'] == 'sac':
                self.episode.clear()
                obs = self.env.reset()
                done = False
                while not done:
                    if policy is None:
                        a = self.env.action_space.sample()
                    else:
                        a = policy.select_action(obs, self.config.agent['explore_noise'])
                    obs_next, r, done, info = self.env.step(a)
                    self.episode.push(obs, a, obs_next, r, done)
                    obs = obs_next
                    results['reward'] += r
                    results['speed'] += info['speed']
                    results['cost'] += info['cost']
                    results['sum_cost'] += info['sum_cost']
                    results['collision_cost'] += info['collision_cost']
                    if info['crash']:
                        results['collision'] = 1
                    results['success'] += info['success']
                    results['fuel'] += info['fuel']
                    results['AEB_num'] += info['AEB_num']
                    # 主进程完成一次sample后，正在sample的子进程结束当前sample
                    if Sub_Proc_Blocking.value == 1:
                        break
                # 主进程正在sample时，结束sample的子进程向主进程发送消息
                if Sub_Proc_Blocking.value == 0:
                    results['time'] = 0.2 * len(self.episode)
                    results['speed'] = results['speed'] / len(self.episode)
                    self.send_lock.acquire()
                    self.send_queue.put((self.id, self.episode, results))
                    self.send_lock.release()
            elif self.config.agent['name'] == 'safe_sac':
                self.episode.clear()
                obs = self.env.reset()
                done = False
                cost = 0
                while not done:
                    if policy is None:
                        a = self.env.action_space.sample()
                    else:
                        a = policy.select_action(obs, self.config.agent['explore_noise'])
                    obs_next, r, done, info = self.env.step(a)
                    cost_next = info['cost']
                    self.episode.push(obs, a, obs_next, r, cost, cost_next, done)
                    obs = obs_next
                    cost = cost_next
                    results['reward'] += r
                    results['speed'] += info['speed']
                    results['cost'] += info['cost']
                    results['sum_cost'] += info['sum_cost']
                    results['collision_cost'] += info['collision_cost']
                    if info['crash']:
                        results['collision'] = 1
                    results['success'] += info['success']
                    results['fuel'] += info['fuel']
                    results['AEB_num'] += info['AEB_num']
                    # 主进程完成一次sample后，正在sample的子进程结束当前sample
                    if Sub_Proc_Blocking.value == 1:
                        break
                # 主进程正在sample时，结束sample的子进程向主进程发送消息
                if Sub_Proc_Blocking.value == 0:
                    results['time'] = 0.2 * len(self.episode)
                    results['speed'] = results['speed'] / len(self.episode)
                    self.send_lock.acquire()
                    self.send_queue.put((self.id, self.episode, results))
                    self.send_lock.release()
            else:
                raise NotImplementedError()


class Sampler():

    def __init__(self, config):
        self.seed = config.seed
        self.config = config
        self.num_workers = config.num_workers

        if config.agent['name'] == 'ppo2':
            Transition = PPO_Transition
        elif (config.agent['name'] == 'cppo') or (config.agent['name'] == 'cppo2'):
            Transition = CPPO_Transition
        elif (config.agent['name'] == 'safe_sac'):
            Transition = Safe_TD_Transition
        else:
            Transition = TD_Transition

        if config.sampler_gpu_index == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda',
                                       index=config.sampler_gpu_index) if torch.cuda.is_available() else torch.device(
                'cpu')

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_workers)])

        self.manager = mp.Manager()
        # Sampler 接受消息的队列
        self.recv_queue = self.manager.Queue(self.num_workers)
        # Sampler 接受消息的队列的锁
        self.recv_lock = self.manager.Lock()

        if is_on_policy(self.config.agent['name']):
            self.buffer = Memory(Transition=Transition)
        else:
            self.buffer = ReplayBuffer(size=config.agent['buffer_size'], Transition=Transition)

        self.workers = [EnvWorker(id, remote, self.recv_queue, self.recv_lock, self.config, Transition)
                        for (id, remote) in zip(range(self.num_workers), self.work_remotes)]

        for worker in self.workers:
            worker.start()

        self.sample_iter = 0
        self.result_dict = {}

    def sample(self, policy):
        self.result_dict = {'reward': [],
                            'speed': [],
                            'time': [],
                            'sum_cost': [],
                            'cost': [],
                            'collision_cost': [],
                            'collision': [],
                            'done': [],
                            'fuel': [],
                            'exploration': [],
                            'action_std': [],
                            'AEB_num': [],
                            }
        Sub_Proc_Blocking.value = 0
        self.buffer.sample_clear()
        if (not self.config.sampler_gpu_index == self.config.agent_gpu_index) and (policy is not None):
            policy = policy.to(self.device)

        for remote in self.remotes:
            remote.send((policy, self.sample_iter))

        # while self.buffer.sample_size < self.args.batch_size:
        while self.buffer.num_episode < self.config.episodes_per_iter:
            id, episode, results = self.recv_queue.get(True)
            self.buffer.push(episode)
            self.result_dict['reward'].append(results['reward'])
            self.result_dict['time'].append(results['time'])
            self.result_dict['speed'].append(results['speed'])
            self.result_dict['fuel'].append(results['fuel'])
            self.result_dict['done'].append(results['success'])
            if 'cost' in results.keys():
                self.result_dict['sum_cost'].append(results['sum_cost'])
                self.result_dict['cost'].append(results['cost'])
                self.result_dict['collision_cost'].append(results['collision_cost'])
            self.result_dict['collision'].append(results['collision'])
            self.result_dict['exploration'] += results['exploration']
            self.result_dict['action_std'] += results['action_std']
            self.result_dict['AEB_num'].append(results['AEB_num'])
            # if self.buffer.sample_size >= self.args.batch_size:
            if self.buffer.num_episode >= self.config.episodes_per_iter:
                Sub_Proc_Blocking.value = 1
                time.sleep(1)
                while not self.recv_queue.empty():
                    self.recv_queue.get()
            else:
                self.remotes[id].send((policy, self.sample_iter))
        self.sample_iter += 1
        if policy is not None:
            self.writer = TensorBoard.get_writer()
            show_train(self.writer, self.config.agent['name'], self.result_dict, self.sample_iter)

        if is_on_policy(self.config.agent['name']):
            return self.buffer.sample()
        else:
            return self.buffer

    def evel(self, policy, evel_episodes=None):
        if evel_episodes == None:
            evel_episodes = self.config.episodes_per_eval
        self.result_dict = {'reward': [],
                            'speed': [],
                            'time': [],
                            'sum_cost': [],
                            'cost': [],
                            'collision_cost': [],
                            'collision': [],
                            'done': [],
                            'fuel': [],
                            'AEB_num': []
                            }
        Sub_Proc_Blocking.value = 0
        if not self.config.sampler_gpu_index == self.config.agent_gpu_index:
            policy = policy.to(self.device)
        for remote in self.remotes:
            remote.send((policy, -1))
        num_episode = 0
        # while self.buffer.sample_size < self.args.batch_size:
        while num_episode < evel_episodes:
            id, results = self.recv_queue.get(True)
            num_episode += 1
            self.result_dict['reward'].append(results['reward'])
            self.result_dict['time'].append(results['time'])
            self.result_dict['speed'].append(results['speed'])
            self.result_dict['fuel'].append(results['fuel'])
            self.result_dict['done'].append(results['success'])
            self.result_dict['AEB_num'].append(results['AEB_num'])
            if 'cost' in results.keys():
                self.result_dict['sum_cost'].append(results['sum_cost'])
                self.result_dict['cost'].append(results['cost'])
                self.result_dict['collision_cost'].append(results['collision_cost'])
            self.result_dict['collision'].append(results['collision'])
            # if self.buffer.sample_size >= self.args.batch_size:
            if num_episode >= evel_episodes:
                Sub_Proc_Blocking.value = 1
                time.sleep(1)
                while not self.recv_queue.empty():
                    self.recv_queue.get()
            else:
                self.remotes[id].send((policy, -1))

        self.writer = TensorBoard.get_writer()
        show_evel(self.writer, self.config.agent['name'], self.result_dict, self.sample_iter)

    def close(self):
        for worker in self.workers:
            if worker.is_alive:
                worker.terminate()
                worker.join()
