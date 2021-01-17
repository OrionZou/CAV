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
parser.add_argument('--reward_model', type=int, default=3,
                    help='reward_model')
parser.add_argument('--human_model', default=False, action="store_true",
                    help='human_model')
parser.add_argument('--gui', default=False, action="store_true",
                    help='gui')
parser.add_argument('--episodes', type=int, default=10,
                    help='evel episodes')
parser.add_argument('--gpu_index', type=int, default=0)
args = parser.parse_args()
config.env['id'] = args.env
config.env['path'] = args.env_path
config.env['human_model'] = args.human_model
config.env['gui'] = args.gui
config.agent_gpu_index = args.gpu_index
np.random.seed(config.seed)
torch.manual_seed(config.seed)
random.seed(config.seed)
torch.cuda.manual_seed(config.seed)

print(vars(config))

if config.agent_gpu_index == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda', index=config.agent_gpu_index) if torch.cuda.is_available() else torch.device('cpu')

env = make_env(config.env, config.seed, 'test', )
model = {
    'CPPO-PID-AEB': ['cppo', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-07/cppo_15-21-35/best_model_i_269.pt'],
    'CPPO-PID': ['cppo', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-08/cppo_14-05-50/best_model_i_153.pt'],
    'PPO-safe-AEB': ['ppo2', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-07/ppo2_15-21-43/best_model_i_252.pt'],
    'PPO-safe': ['ppo2', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-03/ppo2_12-15-18/best_model_i_399.pt'],
    'PPO-AEB': ['ppo2', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-16/ppo2_16-12-47/i_384seed_123_st_687112_ep_7700_tar_2.75.pt'],
    'PPO': ['ppo2', '/home/user/repos/CAV/save_model/expsumo_env-v6_2021-01-03/ppo2_12-15-05/best_model_i_146.pt'],
    'Human': ['human', ''],
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

def smooth(data, weight=0.85):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def evaluation():
    plt.style.use(['science', 'ieee'])
    for i in range(args.episodes):
        print('-----------------', i, '---------------------')
        fig, axes = plt.subplots(2, 1, sharex='col', figsize=(5, 5))
        for key in policys.keys():
            if key == 'Human':
                config.env['human_model'] = True
            else:
                config.env['human_model'] = False
            env = make_env(config.env, seed=i, env_index='test')
            np.random.seed(i)
            torch.manual_seed(i)
            random.seed(i)
            torch.cuda.manual_seed(i)
            print('-----------------', key, '---------------------')
            # env.set_startstep(i)
            for a in range(1):
                obs = env.reset()
                done = False
                step = 0
                distances = []
                speeds = []
                actions=[]
                if key == 'Human':
                    while not done:
                        step += 1
                        distances.append(env.connect.vehicle.getDistance(env.curr_veh_id))
                        speeds.append(env.curr_speed)
                        obs_next, reward, done, info = env.step()
                        if config.env['gui']:
                            time.sleep(0.005)
                else:
                    while not done:
                        step += 1
                        distances.append(env.connect.vehicle.getDistance(env.curr_veh_id))
                        speeds.append(env.curr_speed)

                        obs = torch.Tensor(obs).unsqueeze(0)
                        action = policys[key](obs)
                        action = action.detach().cpu().numpy()[0]
                        obs_next, reward, done, info = env.step(action)
                        obs = obs_next
                        if config.env['gui']:
                            time.sleep(0.005)

            env.close()
            x = np.arange(0, len(distances), 1)
            y = np.array(distances)
            y_1 = np.array(speeds)
            # y_2= np.array(actions)
            axes[0].plot(x, y, label=key)
            axes[1].plot(x, y_1, label=key)
            # axes[2].plot(x, y_2, label=key)
        y = np.ones(150)
        axes[0].fill_between(np.arange(0, 150, 1), 95 * y, 110 * y,
                         facecolor="gray",  # The fill color
                         # color='gray',  # The outline color
                         alpha=0.2, label="Crosswalk")
        axes[0].fill_between(np.arange(0, 150, 1), 140 * y, 145 * y,
                         facecolor="green",  # The fill color
                         # color='green',  # The outline color
                         alpha=0.2, label="End")
        plt.legend(loc='lower right')
        axes[0].set_ylabel('Distance(m)')
        axes[1].set_ylabel('Velocity(m/s)')
        # axes[2].set_ylabel(r'Acceleration($m/s^2$)')
        plt.xlabel('Time(s)')
        # plt.ylabel('Distance(m)')
        plt.legend(loc=8, bbox_to_anchor=(0.4,-0.5),ncol=3)
        plt.show()


if __name__ == "__main__":
    print(vars(args))
    import matplotlib

    print(matplotlib.get_configdir())
    evaluation()
