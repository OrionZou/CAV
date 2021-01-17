from functional import seq
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from common.utils import for_each


class ConstraintModel(torch.nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 ):
        super(ConstraintModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = [32, 32]
        _layer_dims = [self.in_dim] + self.hidden_dim + [self.out_dim]
        self.layers = torch.nn.ModuleList(seq(_layer_dims[:-1])
                                          .zip(_layer_dims[1:])
                                          .map(lambda x: torch.nn.Linear(x[0], x[1]))
                                          .to_list())
        self.last_activation = None

    def _init_weights(self, bound):
        # Initialize all layers except the last one with fan-in initializer
        (seq(self.layers[:-1])
         .map(lambda x: x.weight)
         .for_each(self._initializer))
        # Init last layer with uniform initializer
        torch.nn.init.uniform_(self.layers[-1].weight, -bound, bound)

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = F.relu(layer(out))
        if self.last_activation:
            out = self.last_activation(self.layers[-1](out))
        else:
            out = self.layers[-1](out)
        return out


class SafetyLayer:
    def __init__(self, args, obs_dim,
                 act_dim, num_constraints=1):
        self.args = args
        self.num_constraints = num_constraints
        self.device = torch.device('cpu')
        # if args.gpu_index == -1:
        #     self.device = torch.device('cpu')
        # else:
        #     self.device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device(
        #         'cpu')

        self.models = [ConstraintModel(obs_dim,
                                       act_dim).to(self.device) for _ in range(self.num_constraints)]
        self.optimizers = [Adam(x.parameters(), lr=0.0001) for x in self.models]

    def get_safe_action(self, obs, action, c):
        C = 0.01
        # Find the values of G
        self._eval_mode()
        g = [x(self._as_tensor(obs).view(1, -1)) for x in self.models]
        self._train_mode()
        # Fidn the lagrange multipliers
        g = [x.detach().numpy().reshape(-1) for x in g]
        multipliers = [(np.dot(g_i, action) + c_i-C) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)]
        multipliers = [np.clip(x, 0, np.inf) for x in multipliers]
        # Calculate correction
        correction = np.max(multipliers) * g[np.argmax(multipliers)]
        action_new = action - correction
        return action_new

    def evaluate(self, buffer):
        self._eval_mode()
        for _ in range(5 * buffer.size // self.args.safety_layer_batch_size):
            batch_index = np.random.choice(buffer.size, self.args.safety_layer_batch_size, replace=True)
            batch = buffer.sample(batch_index)
            obs = torch.Tensor(batch.obs).to(self.device)
            action = torch.Tensor(batch.action).to(self.device)
            cost = torch.Tensor(batch.cost).to(self.device)
            cost = cost.reshape(cost.shape[0], 1)
            cost_next = torch.Tensor(batch.cost_next).to(self.device)
            cost_next = cost_next.reshape(cost_next.shape[0], 1)

            for_each(lambda x: x.zero_grad(), self.optimizers)
            gs = [x(obs) for x in self.models]

            c_next_predicted = [cost[:, i] + \
                                torch.bmm(x.view(x.shape[0], 1, -1), action.view(action.shape[0], -1, 1)).view(-1) \
                                for i, x in enumerate(gs)]
            losses = [torch.mean((cost_next[:, i] - c_next_predicted[i]) ** 2) for i in range(self.num_constraints)]
            for_each(lambda x: x.backward(), losses)
            for_each(lambda x: x.step(), self.optimizers)

            if len(losses) > 1:
                loss_avg = np.mean(np.concatenate(np.asarray([x.item() for x in losses])), axis=0)
            else:
                loss_avg = losses[0].item()

        print('Eval:', 'loss_avg', loss_avg, 'losses', losses)

    def train(self, buffer=None):
        self._train_mode()
        for _ in range(5 * buffer.size // self.args.safety_layer_batch_size):
            batch_index = np.random.choice(buffer.size, self.args.safety_layer_batch_size, replace=True)
            batch = buffer.sample(batch_index)
            obs = torch.Tensor(batch.obs).to(self.device)
            action = torch.Tensor(batch.action).to(self.device)
            cost = torch.Tensor(batch.cost).to(self.device)
            cost = cost.reshape(cost.shape[0], 1)
            cost_next = torch.Tensor(batch.cost_next).to(self.device)
            cost_next = cost_next.reshape(cost_next.shape[0], 1)
            for_each(lambda x: x.zero_grad(), self.optimizers)
            gs = [x(obs) for x in self.models]

            c_next_predicted = [cost[:, i] + \
                                torch.bmm(x.view(x.shape[0], 1, -1), action.view(action.shape[0], -1, 1)).view(-1) \
                                for i, x in enumerate(gs)]
            losses = [torch.mean((cost_next[:, i] - c_next_predicted[i]) ** 2) for i in range(self.num_constraints)]
            for_each(lambda x: x.backward(), losses)
            for_each(lambda x: x.step(), self.optimizers)
            if len(losses)>1:
                loss_avg=np.mean(np.concatenate(np.asarray([x.item() for x in losses])), axis=0)
            else:
                loss_avg = losses[0].item()

        print('Train:', 'loss_avg', loss_avg, 'losses', [x.item() for x in losses])

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        return tensor

    def _eval_mode(self):
        for_each(lambda x: x.eval(), self.models)

    def _train_mode(self):
        for_each(lambda x: x.train(), self.models)
