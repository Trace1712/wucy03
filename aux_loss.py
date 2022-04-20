import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import *


def dist(a, b):
    batch = a.shape[0]
    assert batch == b.shape[0]
    return F.pairwise_distance(a.view(batch, -1), b.view(batch, -1))


class LossBase(nn.Module):
    def __init__(self, name, tid, action_space):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.name = name
        self.tid = tid
        self.actiob_space = action_space


class DiverseDynamicLoss(LossBase):
    # """
    # 正向动力学模型辅助任务
    # """
    def __init__(self, name, tid, action_space, state_dim, hidden_size):
        super().__init__(name, tid, action_space)
        self.fc_h = nn.Linear(state_dim + action_space, hidden_size)
        self.fc_z = nn.Linear(hidden_size, state_dim)

    def forward(self, state, action, next_state):
        temp = torch.cat([action, state], dim=1)
        return F.mse_loss(self.fc_z(self.fc_h(temp)), next_state)


class MyInverseDynamicLoss(LossBase):
    # """
    # 逆向动力学模型辅助任务
    # """
    def __init__(self, name, tid, action_space, state_dim, hidden_size):
        super().__init__(name, tid, action_space)
        self.fc_h = nn.Linear(state_dim + action_space, hidden_size)
        self.fc_z = nn.Linear(hidden_size, state_dim)
        self.action_dim = state_dim + action_space

    def forward(self, state, next_state, actions):
        temp = torch.cat([actions, next_state], dim=1)
        temp2 = self.fc_h(temp)
        p_state = self.fc_z(temp2)
        return F.mse_loss(p_state, state)


class InverseDynamicLoss(LossBase):
    def __init__(self, name, tid, action_space, state_dim, hidden_size):
        super().__init__(name, tid, action_space)
        self.fc_h = nn.Linear(state_dim * 2, hidden_size)
        self.fc_z = nn.Linear(hidden_size, action_space)

    def forward(self, feat1, feat2, actions):
        actions, _ = torch.max(actions, dim=1, keepdim=True)
        x = torch.cat([feat1, feat2], dim=1)
        a = self.fc_z(F.relu(self.fc_h(x)))
        return F.cross_entropy(a, actions.squeeze(1).long(), reduction='none')


class RobustRewardLoss(LossBase):
    def __init__(self, name, tid, action_space, state_dim, hidden_size):
        super().__init__(name, tid, action_space)
        self.fc_z = nn.Linear(state_dim, hidden_size)
        self.hc_z = nn.Linear(hidden_size, action_space)

        self.attack_epsilon = 0.075
        self.attack_alpha = 0.0075
        self.attack_iteration = 2

    def forward(self, state, action):
        # ori_state = self.normalize(state.data)
        ori_state_tensor = torch.tensor(state.clone().detach(), dtype=torch.float32)
        # random start
        criterion = torch.nn.MSELoss()
        state = torch.tensor(state, dtype=torch.float32, requires_grad=True)

        for i in range(self.attack_iteration):
            gt_action = self.hc_z(self.fc_z(state))
            loss = -criterion(action, gt_action)
            self.zero_grad()
            state.register_hook(extract)
            loss.requires_grad_(True)
            loss.backward()
            adv_state = state - self.attack_alpha * torch.sign(torch.tensor(features_grad))
            state = torch.min(torch.max(adv_state, ori_state_tensor - self.attack_epsilon),
                              ori_state_tensor + self.attack_epsilon)

        return F.mse_loss(gt_action, action)

    def to_np(self, t):
        return t.cpu().detach().numpy()


class MomentChangesLoss(LossBase):
    def __init__(self, name, tid, args, action_space, state_dim, hidden_size):
        super().__init__(name, tid, args, action_space)

        self.lstm = nn.LSTM(state_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, action_space)

        # 初始化隐含状态及细胞状态C，hidden_cell变量包含先前的隐藏状态和单元状态
        self.hidden_cell = (torch.zeros(1, 1, hidden_size),
                            torch.zeros(1, 1, hidden_size))

        self.fc_h = nn.Linear(args.state_dim + 10 + action_space, args.hidden_size)
        self.fc_z = nn.Linear(args.hidden_size, args.state_dim + 10)

    def forward(self, state, actions, next_state):
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # # lstm的输出是当前时间步的隐藏状态ht和单元状态ct以及输出lstm_out
        # # 按照lstm的格式修改input_seq的形状，作为linear层的输入
        # predictions = self.linear(lstm_out)
        # return predictions[-1]  # 返回predictions的最后一个元素
        # temp = torch.cat([actions, next_state], dim=1)
        # return F.mse_loss(self.fc_z(self.fc_h(temp)), state)
        pass


class RewardAttackLoss(LossBase):

    def __init__(self, name, tid, action_space, state_dim, hidden_size):
        super().__init__(name, tid, action_space)
        self.fc_z = nn.Linear(state_dim + action_space, hidden_size)
        self.hc_z = nn.Linear(hidden_size, 1)
        self.gamma = 0.99
        self.reward_scale = 1

    def forward(self, critic_value, rewards, state, actions, dis_critic_loss):
        temp = torch.cat([actions, state], dim=1)
        aux_critic_loss = self.hc_z(self.fc_z(temp))
        y = self.reward_scale * rewards * (-1) + self.gamma * (
            aux_critic_loss - dis_critic_loss
        )
        return F.mse_loss(y, critic_value)


def get_loss_by_name(name):
    if name == 'InverseDynamicLoss':
        # 预测action（正向任务）
        return InverseDynamicLoss
    # elif name == 'MomentChanges':
    #     # 瞬时变化奖励（鲁棒任务）
    #     return MomentChangesLoss
    elif name == "MyInverseDynamicLoss":
        # 逆向动力学模型（正向任务）
        return MyInverseDynamicLoss
    elif name == "DiverseDynamicLoss":
        # 正向动力学模型（正向任务）
        return DiverseDynamicLoss
    elif name == "RewardAttackLoss":
        # 模型攻击奖励（鲁棒任务）
        return RewardAttackLoss
    elif name == "RobustRewardLoss":
        # 模型攻击状态（鲁棒任务）
        return RobustRewardLoss


def get_aux_loss(name, *args):
    return get_loss_by_name(name)(name, *args)


def extract(g):
    global features_grad
    features_grad = g
