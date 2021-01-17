import os
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sh = True
if sh:
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(os.path.split(rootPath)[0])
    proj_path = os.path.abspath(os.path.join(os.getcwd()))
else:
    proj_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

import time
import argparse
import datetime
import torch
import numpy as np
import random
from agents import *
from torch.utils.tensorboard import SummaryWriter
from envs.naive_v6_1_1.sampler import Sampler, make_env
from common.utils import write_config
from common.tensorboard import TensorBoard, show_agent
from common.networks import *
from common.buffers import *

import envs.naive_v6_1_1
import envs.multi_lane_v1
from envs.naive_v6_1_1.config import Config

config = Config()

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in sumo')
parser.add_argument('--env', type=str, default='sumo_env-v6',
                    help=' environment')
if sh:
    parser.add_argument('--env_path', type=str, default=sys.path[0] + '/exp.sumocfg',
                        help=' environment scenario path')
else:
    parser.add_argument('--env_path', type=str, default='exp.sumocfg',
                        help=' environment scenario path')
parser.add_argument('--reward_model', type=int, default=2,
                    help='reward_model')
parser.add_argument('--human_model', default=False, action="store_true",
                    help='human_model')
parser.add_argument('--gui', default=False, action="store_true",
                    help='gui')
parser.add_argument('--seed', type=int, default=0,
                    help='seed for random number generators')

parser.add_argument('--algo', type=str, default='ppo2',
                    help=' algorithm cppo ppo2 ddpg td3')
parser.add_argument('--load_path', type=str, default=None,
                    help='rl model path')
parser.add_argument('--episodes', type=int, default=60,
                    help='evel episodes')
parser.add_argument('--gpu_index', type=int, default=0)

# parser.add_argument('--model_path', type=str, default='tests/save_model/sumo_sac_s_0_ep_3650_tr_-6.29_er_-9.2.pt')

args = parser.parse_args()
config.env['id'] = args.env
config.env['path'] = args.env_path
config.env['reward_model'] = args.reward_model
config.env['human_model'] = args.human_model
config.env['gui'] = args.gui
config.seed = args.seed
config.agent_gpu_index = args.gpu_index
config.select_agent(args.algo)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
random.seed(config.seed)
torch.cuda.manual_seed(config.seed)

if config.agent_gpu_index == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda', index=config.agent_gpu_index) if torch.cuda.is_available() else torch.device('cpu')

curr_time = datetime.datetime.now().strftime("%H-%M-%S")
curr_day = datetime.datetime.now().strftime("%Y-%m-%d")
# init save model
save_path = config.create_save_path()
write_config(save_path, config)

print(vars(config))

# env = make_env(config.env, config.seed, 'test', )
#
# policys = {}
# if args.algo == 'cppo':
#     policys[args.algo] = CPPOModel(env.observation_space.shape[0], env.action_space.shape[0])
# elif args.algo == 'ppo2':
#     policys[args.algo] = PPOModel(env.observation_space.shape[0], env.action_space.shape[0])
# elif args.algo == 'ddpg' or args.algo == 'td3':
#     policys[args.algo] = DDPGActor(env.observation_space.shape[0], env.action_space.shape[0],
#                                    hidden_sizes=(128, 128),
#                                    output_activation=torch.tanh)
# elif args.algo == 'human':
#     pass
# else:
#     policys[args.algo] = SACActor(env.observation_space.shape[0], env.action_space.shape[0],
#                                   hidden_sizes=(64, 64), )
#
# if not args.algo == 'human':
#     policys[args.algo].load_state_dict(torch.load(args.load_path, map_location=device))


def slope(distances):
    distances = np.array(distances)
    return (distances[1:] - distances[:-1])


env = make_env(config.env, config.seed, 'test', )
model = {
    '(a) CPPO-PID-AEB': ['cppo', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-07/cppo_15-21-35/best_model_i_269.pt'],
    '(b) CPPO-PID': ['cppo', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-08/cppo_14-05-50/best_model_i_153.pt'],
    '(c) PPO-safe-AEB': ['ppo2', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-07/ppo2_15-21-43/best_model_i_252.pt'],
    '(d) PPO-safe': ['ppo2', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-03/ppo2_12-15-18/best_model_i_399.pt'],
    '(e) PPO-AEB': ['ppo2', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-16/ppo2_16-12-47/i_384seed_123_st_687112_ep_7700_tar_2.75.pt'],
    '(f) PPO': ['ppo2', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-03/ppo2_12-15-05/best_model_i_146.pt'],
    '(g) Human': ['human', ''],
}
policys = {}

for key in model.keys():
    algo_name = model[key][0]
    if algo_name == 'cppo':
        policys[key] = CPPOModel(env.observation_space.shape[0], env.action_space.shape[0])
        policys[key].load_state_dict(torch.load(model[key][1], map_location=device))
    elif algo_name == 'ppo2':
        policys[key] = PPOModel(env.observation_space.shape[0], env.action_space.shape[0])
        policys[key].load_state_dict(torch.load(model[key][1], map_location=device))
    else:
        policys[key] = None
env.close()

def evaluation(algo_name,ax):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    if algo_name == '(g) Human':
        config.env['human_model'] = True
    env = make_env(config.env, config.seed, 'test', )
    # fig, axs = plt.subplots(figsize=(3, 1))
    # segments = np.full((args.episodes, config.env['max_step'], 2), np.nan)
    # dydx_s = np.full((args.episodes, config.env['max_step']), np.nan)
    segments = []
    dydx_s = []
    for i in range(args.episodes):
        results = {}
        results['time'] = 0
        results['speed'] = 0
        results['collision'] = 0
        results['success'] = 0
        results['fuel'] = 0
        print('-----------------', i, '---------------------')
        # env.set_startstep(i)
        obs = env.reset()
        done = False
        speed = 0
        step = 0
        distances = []
        if algo_name == '(g) Human':
            while not done:
                step += 1
                distances.append(env.connect.vehicle.getDistance(env.curr_veh_id))
                obs_next, reward, done, info = env.step()
                speed += info['speed']
                results['collision'] += info['crash']
                results['success'] += info['success']
                results['fuel'] += info['fuel']
                # print(info['speed'])

        else:
            while not done:
                step += 1
                distances.append(env.connect.vehicle.getDistance(env.curr_veh_id))
                obs = torch.Tensor(obs).unsqueeze(0)
                action = policys[algo_name](obs)
                action = action.detach().cpu().numpy()[0]
                obs_next, reward, done, info = env.step(action)
                obs = obs_next

                speed += info['speed']
                results['collision'] += info['crash']
                results['success'] += info['success']
                results['fuel'] += info['fuel']

            if config.env['gui']:
                time.sleep(0.005)
        results['time'] += 0.2 * step
        results['speed'] += speed / step

        dydx = slope(distances)
        # distances = distances[:-1]
        x = np.arange(0 + i * 40, len(distances) + i * 40, 1)
        y = np.array(distances)
        dydx_s.append(dydx)
        segments.append(np.array([x, y]).T.reshape(-1, 2))
        # dydx_s[i, :x.shape[0]] = dydx
        # segments[i, :x.shape[0], :] = np.array([x, y]).T.reshape(-1, 2)
        print(results)
    # dydx_s = dydx_s.T.reshape(-1)
    # dydx_s = dydx_s[~np.isnan(dydx_s)]
    # dydx_s=dydx_s.reshape(-1)
    norm = plt.Normalize(0, 4)
    for i in range(args.episodes):
        segment = np.concatenate([[segments[i][:-1, :]], [segments[i][1:, :]]], axis=0)
        segment = segment.transpose(1, 0, 2)
        lc = LineCollection(segment, cmap='viridis', norm=norm)
        lc.set_array(dydx_s[i])
        lc.set_linewidth(0.7)
        line = ax.add_collection(lc)
    # lc = LineCollection(segments, cmap='viridis', norm=norm)

    cb = fig.colorbar(line, ax=ax, )
    cb.set_label('speed(m/0.2s)')
    x = np.arange(0, len(distances) + i * 40, 1)
    y = np.ones(len(distances) + i * 40)
    plt.fill_between(x, 95 * y, 110 * y,
                     facecolor="gray",  # The fill color
                     # color='gray',  # The outline color
                     alpha=0.2,label="Crosswalk")
    plt.fill_between(x, 140 * y, 145 * y,
                     facecolor="green",  # The fill color
                     # color='green',  # The outline color
                     alpha=0.2,label="End")
    # ax.fill_between(x, 95 * y, 110 * y,
    #                  facecolor="gray",  # The fill color
    #                  # color='gray',  # The outline color
    #                  alpha=0.2, )
    # ax.fill_between(x, 140 * y, 145 * y,
    #                  facecolor="green",  # The fill color
    #                  # color='green',  # The outline color
    #                  alpha=0.2, )

    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Distance(m)')
    plt.legend(loc='lower right')
    # ax.set_title(algo_name)
    env.close()



if __name__ == "__main__":
    print('Evaluation:', args.algo)
    print(vars(args))
    plt.style.use(['science', 'ieee'])
    fig, axes = plt.subplots(1, 1, sharex='col', figsize=(4, 2.2))
    evaluation('(g) Human', axes)
    # a=0
    # for key in policys.keys():
    #     evaluation(key,axes[a])
    #     a+=1
    #     if a==4:
    #         break
    plt.show()
