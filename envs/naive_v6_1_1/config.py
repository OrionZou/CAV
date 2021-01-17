import os
import sys
import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
proj_path = os.path.abspath(os.path.join(os.getcwd()))

class Config:

    def __init__(self):
        self.env = {
            'id': 'sumo_env-v6',
            'path': sys.path[0] + '/exp.sumocfg',
            'gui': False,
            'max_step': 150,
            'reward_model': 3,
            '': False,
        }
        self.seed = 0
        self.num_workers = 6
        self.episodes_per_iter = 100
        self.episodes_per_eval = 100
        self.iterations = 400
        self.agent = {
            'name': 'ppo2',
            'gamma': 0.995,
            'lamda': 0.97,
            'lr': 3E-4,
            'minibatch_size': 256,
            'num_epoch': 10,
            'loss_coeff_entropy': 0.01,
            'load_path': None
        }
        self.sampler_gpu_index = -1
        self.agent_gpu_index = 0

    def select_agent(self, agent_name):
        if agent_name == 'cppo':
            self.agent = {
                'name': 'cppo',
                'gamma': 0.995,
                'lamda': 0.97,
                'cost_gamma': 0.99,
                'cost_lambda': 0.97,
                'lr': 3E-4,
                'minibatch_size': 256,
                'loss_coeff_entropy': 0.01,
                'load_path': None,
                'cost_limit': 0.0001,
                'penalty_init': 1,
                'kp': 1,
                'ki': 0.01,
                'kd': 4,
            }
        elif agent_name in ['sac']:
            self.agent = {
                'name': 'sac',
                'gamma': 0.99,
                'lr': 3E-4,
                'explore_noise':0.1,
                'minibatch_size': 256,
                'reward_scale': 10.,
                'num_epoch': 2 ** 0,
                'buffer_size': 2 ** 17,
                'load_path': None,
            }
        elif agent_name in ['safe_sac']:
            self.agent = {
                'name': 'safe_sac',
                'gamma': 0.99,
                'lr': 3E-4,
                'explore_noise':0.1,
                'minibatch_size': 256,
                'reward_scale': 10.,
                'num_epoch': 2 ** 0,
                'buffer_size': 2 ** 17,
                'load_path': None,
            }
        else:
            self.agent = {
                'name': 'ppo2',
                'gamma': 0.995,
                'lamda': 0.97,
                'lr': 3E-4,
                'minibatch_size': 256,
                'loss_coeff_entropy': 0.01,
                'load_path': None
            }

    def create_save_path(self):

        curr_time = datetime.datetime.now().strftime("%H-%M-%S")
        curr_day = datetime.datetime.now().strftime("%Y-%m-%d")

        # init save model
        if not os.path.exists(proj_path + '/save_model'):
            os.mkdir(proj_path + '/save_model')
        if not os.path.exists(proj_path + '/save_model/exp' + self.env['id'] + '_' + curr_day):
            os.mkdir(proj_path + '/save_model/exp' + self.env['id'] + '_' + curr_day)
        if not os.path.exists(proj_path + '/save_model/exp' + self.env['id'] + '_' + curr_day + '/'
                              + self.agent['name'] + '_' + curr_time):
            os.mkdir(proj_path + '/save_model/exp' + self.env['id'] + '_' + curr_day + '/'
                     + self.agent['name'] + '_' + curr_time)
        save_path = proj_path + '/save_model/exp' + self.env['id'] + '_' + curr_day + '/' \
                    + self.agent['name'] + '_' + curr_time
        self.save_path = save_path
        return save_path

# if __name__ == '__main__':
#     c = Config()
#     print(vars(c))
