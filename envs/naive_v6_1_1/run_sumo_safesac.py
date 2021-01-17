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
import envs
import time
import argparse
import datetime
import torch
import random
from torch.utils.tensorboard import SummaryWriter
from envs.naive_v6_1_1.sampler import Sampler
import gym
from common.vis import *
from common.utils import write_config
from safety_layer.safety_layer import SafetyLayer

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--env', type=str, default='sumo_env-v6',
                    help=' environment')
if sh:
    parser.add_argument('--env_path', type=str, default=sys.path[0] + '/exp.sumocfg',
                        help=' environment scenario path')
else:
    parser.add_argument('--env_path', type=str, default='exp.sumocfg',
                        help=' environment scenario path')

parser.add_argument('--env_gui', type=bool, default=False,
                    help='environment gui')
parser.add_argument('--env_max_step', type=int, default=150,
                    help='environment max step')
parser.add_argument('--seed', type=int, default=0,
                    help='seed for random number generators')

parser.add_argument('--num_workers', type=int, default=4,
                    help='workers for sample')

parser.add_argument('--explore_noise', type=float, default=0.1,
                    help='exploration noise stop iteration')

parser.add_argument('--iterations', type=int, default=2000,
                    help='iterations number')

parser.add_argument('--algo', type=str, default='safe_sac',
                    help='select an algorithm among trpo, ppo2, ddpg, td3, sac')
parser.add_argument('--reward_scale', type=int, default=20.,
                    help='reward_scale')

parser.add_argument('--episode_size', type=int, default=20,
                    help='episode_size')
parser.add_argument('--evel_episode_size', type=int, default=100,
                    help='episode_size')
parser.add_argument('--max_step', type=int, default=2 ** 10,
                    help='max step for environment')
# parser.add_argument('--batch_size', type=int, default=2 ** 10,
#                     help='max step for environment')
parser.add_argument('--minibatch_size', type=int, default=2 ** 8,
                    help='buffer_size')
parser.add_argument('--repeat_times', type=int, default=2 ** 0,
                    help='repeat_times')
parser.add_argument('--buffer_size', type=int, default=2 ** 17,
                    help='buffer_size')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='rl agent gamma')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='rl agent lr')

parser.add_argument('--safety_layer_epoch', type=int, default=100,
                    help='safety_layer_epoch')
parser.add_argument('--safety_layer_batch_size', type=int, default=256,
                    help='safety_layer_epoch')
parser.add_argument('--sample_scale', type=int, default=10,
                    help='sample_scale')

parser.add_argument('--load', type=str, default=None,
                    help='copy & paste the saved model name, and load it')
parser.add_argument('--train_model', type=bool, default=True,
                    help='train model is Ture. eval model is False')
parser.add_argument('--gpu_index', type=int, default=1)

# parser.add_argument('--model_path', type=str, default='tests/save_model/sumo_sac_s_0_ep_3650_tr_-6.29_er_-9.2.pt')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.gpu_index == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

begin_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# init save model
if not os.path.exists(proj_path + '/save_model'):
    os.mkdir(proj_path + '/save_model')
if not os.path.exists(proj_path + '/save_model/' + args.env):
    os.mkdir(proj_path + '/save_model/' + args.env)
if not os.path.exists(proj_path + '/save_model/' + args.env + '/' + args.algo + '-' + begin_time):
    os.mkdir(proj_path + '/save_model/' + args.env + '/' + args.algo + '-' + begin_time)
save_path = proj_path + '/save_model/' + args.env + '/' + args.algo + '-' + begin_time
write_config(save_path, args)

# init tensorboard
writer = SummaryWriter(log_dir=save_path)
print(vars(args))

if args.algo == 'safe_sac':
    from agents.sac import Agent


def main():
    env = gym.make(args.env)
    print('State dimension:', env.observation_space.shape[0])
    print('Action dimension:', env.action_space.shape[0])
    print('Action limited:', env.action_space.low, "~", env.action_space.high)

    memorySampler = Sampler(args)
    safety_layer = SafetyLayer(args, env.observation_space.shape[0], env.action_space.shape[0])

    # 载入网络参数
    if args.load is not None:
        pretrained_model_path = os.path.join(proj_path + '/save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path, map_location=device)
        agent.policy.load_state_dict(pretrained_model)

    finished_steps = 0
    finished_episodes = 0
    best_target = 0
    save_index = 0
    print('____________________train safety layer_________________________')
    for i in range(args.safety_layer_epoch):
        for _ in range(args.sample_scale):
            memorySampler.sample(None)
        safety_layer.train(memorySampler.buffer)
        memorySampler.buffer.clear()
    for _ in range(args.sample_scale):
        memorySampler.sample(None)
    safety_layer.evaluate(memorySampler.buffer)
    memorySampler.buffer.clear()
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], args, device, hidden_sizes=(64, 64))
    print('____________________initial exploration_________________________')
    agent.policy.set_action_modifier(safety_layer.get_safe_action)
    for i in range(5):
        memorySampler.sample(None)
    print('steps:', memorySampler.buffer.size)
    print('____________________train agent_________________________')
    for i in range(args.iterations):
        print('________________________iter:', i, '_____________________________')
        '''sample stage'''
        start_time = time.time()
        samples = memorySampler.sample(agent.policy)
        sample_time = time.time() - start_time
        '''train stage'''
        start_time = time.time()
        agent.train_model(samples)
        train_time = time.time() - start_time
        print('pi_loss', agent.record_policy_loss, 'q_loss', agent.record_qf_loss)
        print('TRAIN:')
        print('reward: mean', np.mean(memorySampler.result_dict['reward']), 'max',
              np.max(memorySampler.result_dict['reward']), 'min', np.min(memorySampler.result_dict['reward']), 'std',
              np.std(memorySampler.result_dict['reward']))
        print('cost: mean', np.mean(memorySampler.result_dict['cost']), 'max',
              np.max(memorySampler.result_dict['cost']), 'min', np.min(memorySampler.result_dict['cost']), 'std',
              np.std(memorySampler.result_dict['cost']))
        print('done rate:', np.mean(memorySampler.result_dict['done']), 'collision rate:',
              np.mean(memorySampler.result_dict['collision']))
        '''for saving model'''
        finished_steps += memorySampler.buffer.sample_size
        finished_episodes += memorySampler.buffer.num_episode
        target = np.mean(memorySampler.result_dict['reward']) / np.mean(memorySampler.result_dict['time'])

        show_agent(writer, args.algo, agent, i)
        show_train(writer, args.algo, memorySampler.result_dict, i)
        '''evel stage'''
        start_time = time.time()
        memorySampler.evel(agent.policy)
        evel_time = time.time() - start_time
        show_evel(writer, args.algo, memorySampler.result_dict, i)
        print('EVELUATION:')
        print('reward: mean', np.mean(memorySampler.result_dict['reward']), 'max',
              np.max(memorySampler.result_dict['reward']), 'min', np.min(memorySampler.result_dict['reward']),
              'std',
              np.std(memorySampler.result_dict['reward']))
        print('cost: mean', np.mean(memorySampler.result_dict['cost']), 'max',
              np.max(memorySampler.result_dict['cost']), 'min', np.min(memorySampler.result_dict['cost']), 'std',
              np.std(memorySampler.result_dict['cost']))
        print('done rate:', np.mean(memorySampler.result_dict['done']), 'collision rate:',
              np.mean(memorySampler.result_dict['collision']))

        print('USE TIME AND SAMPLES:')
        # print('sample time:', sample_time, 'train time:',
        #       train_time, 'steps:', memorySampler.buffer.sample_size,
        #       'total steps:', finished_steps, 'episodes:',
        #       memorySampler.buffer.num_episode, 'total episodes:', finished_episodes)
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
