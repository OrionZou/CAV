import os
import sys

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
from torch.utils.tensorboard import SummaryWriter
from envs.naive_v6_1_1.sampler import Sampler, make_env
from common.utils import write_config
from common.tensorboard import TensorBoard, show_agent
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

parser.add_argument('--seed', type=int, default=0,
                    help='seed for random number generators')

parser.add_argument('--num_workers', type=int, default=6,
                    help='workers for sample')
parser.add_argument('--episodes_per_iter', type=int, default=20,
                    help='episodes_per_iter')

parser.add_argument('--algo', type=str, default='ppo2',
                    help='select an algorithm among trpo, ppo2, ddpg, td3, sac')
# parser.add_argument('--gamma', type=float, default=0.995,
#                     help='rl agent gamma')
# parser.add_argument('--lamda', type=float, default=0.97,
#                     help='rl agent lamda')
# parser.add_argument('--lr', type=float, default=3e-4,
#                     help='rl agent lr')
# parser.add_argument('--loss_coeff_entropy', type=float, default=0.01,
#                     help='loss_coeff_entropy')

parser.add_argument('--load', type=str, default=None,
                    help='copy & paste the saved model name, and load it')
parser.add_argument('--gpu_index', type=int, default=0)

# parser.add_argument('--model_path', type=str, default='tests/save_model/sumo_sac_s_0_ep_3650_tr_-6.29_er_-9.2.pt')

args = parser.parse_args()
config.gpu_index = args.gpu_index
config.env['id'] = args.env
config.env['path'] = args.env_path
config.env['reward_model'] = args.reward_model
config.env['human_model'] = args.human_model
config.seed = args.seed
config.episodes_per_iter = args.episodes_per_iter
config.num_workers = args.num_workers
config.agent_gpu_index = args.gpu_index
config.select_agent(args.algo)

np.random.seed(config.seed)
torch.manual_seed(config.seed)
random.seed(config.seed)
torch.cuda.manual_seed(config.seed)

if config.gpu_index == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda', index=config.agent_gpu_index) if torch.cuda.is_available() else torch.device('cpu')

curr_time = datetime.datetime.now().strftime("%H-%M-%S")
curr_day = datetime.datetime.now().strftime("%Y-%m-%d")
# init save model
save_path = config.create_save_path()
write_config(save_path, config)

print(vars(config))
# init tensorboard
writer = TensorBoard.get_writer(load_path=save_path)

if config.agent['name'] == 'ppo2':
    from agents.ppo2 import Agent
elif config.agent['name'] == 'cppo':
    from agents.cppo import Agent
elif config.agent['name'] == 'sac':
    from agents.sac import Agent


def main():
    env = make_env(config.env, config.seed, -1)
    print('State dimension:', env.observation_space.shape[0])
    print('Action dimension:', env.action_space.shape[0])
    print('Action limited:', env.action_space.low, "~", env.action_space.high)

    memorySampler = Sampler(config)
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], config)

    # 载入网络参数
    if args.load is not None:
        pretrained_model_path = os.path.join(proj_path + '/save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path, map_location=device)
        agent.policy.load_state_dict(pretrained_model)

    finished_steps = 0
    finished_episodes = 0
    best_target = 0
    save_index = 0

    for i in range(config.iterations):
        print('________________________iter:', i, '_____________________________')
        '''sample stage'''
        start_time = time.time()
        samples = memorySampler.sample(agent.policy)
        sample_time = time.time() - start_time

        '''train stage'''
        start_time = time.time()
        agent.train_model(samples, iter_index=i)
        train_time = time.time() - start_time
        show_agent(writer, config.agent['name'], agent, i)

        if config.agent['name'] in ['cppo', 'ppo', 'sac']:
            agent.print_loss()
        else:
            print('actor_loss', agent.record_policy_loss, 'critic_loss', agent.record_vf_loss, 'entropy',
                  agent.record_entropy_loss)
        print('TRAIN:')
        print('reward: mean', np.mean(memorySampler.result_dict['reward']), 'max',
              np.max(memorySampler.result_dict['reward']), 'min',
              np.min(memorySampler.result_dict['reward']), 'std', np.std(memorySampler.result_dict['reward']))
        print('cost: mean', np.mean(memorySampler.result_dict['cost']), 'max',
              np.max(memorySampler.result_dict['cost']), 'min',
              np.min(memorySampler.result_dict['cost']), 'std', np.std(memorySampler.result_dict['cost']))
        print('done rate:', np.mean(memorySampler.result_dict['done']), 'collision rate:',
              np.mean(memorySampler.result_dict['collision']))
        '''for saving model'''
        finished_steps += len(memorySampler.buffer)
        finished_episodes += memorySampler.buffer.num_episode
        target = np.mean(memorySampler.result_dict['reward']) / np.mean(memorySampler.result_dict['time'])

        # '''evel stage'''
        start_time = time.time()
        memorySampler.evel(agent.policy)
        evel_time = time.time() - start_time
        print('EVELUATION:')
        print('reward: mean', np.mean(memorySampler.result_dict['reward']), 'max',
              np.max(memorySampler.result_dict['reward']), 'std', np.std(memorySampler.result_dict['reward']))
        print('cost: mean', np.mean(memorySampler.result_dict['cost']), 'max',
              np.max(memorySampler.result_dict['cost']), 'std', np.std(memorySampler.result_dict['cost']))
        print('done rate:', np.mean(memorySampler.result_dict['done']), 'collision rate:',
              np.mean(memorySampler.result_dict['collision']))

        # print('USE TIME AND SAMPLES:')
        # print('sample time:', sample_time, 'train time:',
        #       train_time, 'steps:', memorySampler.buffer.sample_size,
        #       'total steps:', finished_steps, 'episodes:',
        #       memorySampler.buffer.num_episode, 'total episodes:', finished_episodes)

        print('USE TIME AND SAMPLES:')
        print('sample time:', sample_time, 'train time:',
              train_time, 'evel time:', evel_time, 'steps:', memorySampler.buffer.sample_size,
              'total steps:', finished_steps, 'episodes:',
              memorySampler.buffer.num_episode, 'total episodes:', finished_episodes)

        # Save the trained model
        condition = 2
        # 保存满足条件的模型
        if target > condition:
            ckpt_path = os.path.join(save_path \
                                     + '/i_' + str(i) \
                                     + 'seed_' + str(args.seed) \
                                     + '_st_' + str(finished_steps) \
                                     + '_ep_' + str(finished_episodes) \
                                     + '_tar_' + str(round(target, 2)) + '.pt')
            torch.save(agent.policy.state_dict(), ckpt_path)
            if target > best_target:
                if save_index > 0:
                    os.remove(save_path + '/best_model_i_' + str(save_index) + '.pt')
                best_ckpt_path = os.path.join(save_path \
                                              + '/best_model_i_' + str(i) + '.pt')
                save_index = i
                torch.save(agent.policy.state_dict(), best_ckpt_path)
                best_target = target
    memorySampler.close()


if __name__ == '__main__':
    main()
