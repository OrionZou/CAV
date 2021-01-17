from tensorboardX import SummaryWriter
import numpy as np


class TensorBoard:
    _writer = None

    @classmethod
    def get_writer(cls, load_path=None):
        if cls._writer:
            return cls._writer
        cls._writer = SummaryWriter(load_path)
        return cls._writer


def show_agent(writer, algo_name, agent, iter):
    if algo_name == 'ppo2' or algo_name == 'cppo2':
        writer.add_scalar('Algo/PolicyLoss', agent.record_policy_loss, iter)
        writer.add_scalar('Algo/ValueLoss', agent.record_vf_loss, iter)
        writer.add_scalar('Algo/EntropyLoss', agent.record_entropy_loss, iter)
    elif algo_name == 'cppo':
        writer.add_scalar('Algo/PolicyLoss', agent.record_policy_loss, iter)
        writer.add_scalar('Algo/RValueLoss', agent.record_vf_r_loss, iter)
        writer.add_scalar('Algo/CValueLoss', agent.record_vf_c_loss, iter)
        writer.add_scalar('Algo/EntropyLoss', agent.record_entropy_loss, iter)
        writer.add_scalar('Algo/Cost_penalty', agent.cost_penalty, iter)
    elif algo_name == 'td3':
        writer.add_scalar('Algo/PolicyLoss', np.mean(agent.policy_losses), iter)
        writer.add_scalar('Algo/ValueLoss', np.mean(agent.qf_losses), iter)
    elif algo_name == 'sac' or algo_name == 'safe_sac':
        writer.add_scalar('Algo/PolicyLoss', agent.record_policy_loss, iter)
        writer.add_scalar('Algo/ValueLoss', agent.record_vf_loss, iter)
        writer.add_scalar('Algo/AlphaLoss', agent.record_alpha_loss, iter)


def show_train(writer, algo_name, result_dict, iter):
    algo_list = ['ppo2', 'cppo', 'cppo2', 'sac', 'td3', 'safe_sac']
    if algo_name in algo_list:
        if not algo_name == ['sac','safe_sac']:
            writer.add_scalar('Train/MaxActionExplor', np.max(result_dict['exploration']), iter)
            writer.add_scalar('Train/MeanActionExplor', np.mean(result_dict['exploration']), iter)
            writer.add_scalar('Train/MeanActionStd', np.mean(result_dict['action_std']), iter)

        writer.add_scalar('Train/DoneRate', np.mean(result_dict['done']), iter)
        writer.add_scalar('Train/CollisionRate', np.mean(result_dict['collision']), iter)
        writer.add_scalar('Train/AverageReturn', np.mean(result_dict['reward']), iter)
        writer.add_scalar('Train/AverageCost', np.mean(result_dict['cost']), iter)
        writer.add_scalar('Train/AverageReturnsTime', np.mean(result_dict['reward']) / np.mean(result_dict['time']),
                          iter)
        writer.add_scalar('Train/AverageTime', np.mean(result_dict['time']), iter)
        writer.add_scalar('Train/AverageSpeed', np.mean(result_dict['speed']), iter)
        writer.add_scalar('Train/AverageFuel', np.mean(result_dict['fuel']), iter)
        writer.add_scalar('Train/AverageAEB', np.mean(result_dict['AEB_num']), iter)
        #
        # writer.add_scalar('Train/StdReturn', np.std(result_dict['reward']), iter)
        # writer.add_scalar('Train/Stdtime', np.std(result_dict['time']), iter)


def show_evel(writer, algo_name, result_dict, iter):
    algo_list = ['ppo2', 'cppo', 'cppo2', 'sac', 'td3', 'safe_sac']
    if algo_name in algo_list:
        writer.add_scalar('Evel/DoneRate', np.mean(result_dict['done']), iter)
        writer.add_scalar('Evel/CollisionRate', np.mean(result_dict['collision']), iter)
        writer.add_scalar('Evel/AverageReturn', np.mean(result_dict['reward']), iter)
        writer.add_scalar('Evel/AverageCost', np.mean(result_dict['cost']), iter)
        writer.add_scalar('Evel/AverageReturnsTime', np.mean(result_dict['reward']) / np.mean(result_dict['time']),
                          iter)
        writer.add_scalar('Evel/AverageTime', np.mean(result_dict['time']), iter)
        writer.add_scalar('Evel/AverageSpeed', np.mean(result_dict['speed']), iter)
        writer.add_scalar('Evel/AverageFuel', np.mean(result_dict['fuel']), iter)
        writer.add_scalar('Evel/AverageAEB', np.mean(result_dict['AEB_num']), iter)
        #
        # writer.add_scalar('Evel/StdReturn', np.std(result_dict['reward']), iter)
        # writer.add_scalar('Evel/Stdtime', np.std(result_dict['time']), iter)
