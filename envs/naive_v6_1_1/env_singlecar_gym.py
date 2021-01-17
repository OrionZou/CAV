import os
import sys
from envs.sumo_util import OBB, isCollision, clac_center_point

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import numpy as np
import gym
import math
import torch
import random
from gym import spaces
import time
import envs.naive_v6_1_1
from sumolib import checkBinary


class SUMO_ENV(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, path="exp.sumocfg", gui=False, label='naive', max_step=150):
        '''环境动态的静态变量'''
        self.__actionStepLength = 0.2
        self.__veh_state_num = 6
        self.__ped_state_num = 5
        self.__accel = 3.0
        self.__decel = 4.0
        self.__AEBS_brake = 8
        self.__AEBS_state = 0
        self.__MAX_PERSON_COUNT = 4  # 最大观测行人数MAX_PERSON_COUNT
        self.__MAX_SPEED = 0
        '''公开变量'''
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1e3, high=1e3, shape=(
            self.__veh_state_num + self.__ped_state_num * self.__MAX_PERSON_COUNT,))

        '''启动环境设定'''
        self.gui = gui
        self.path = path
        self.label = label
        self.max_step = max_step
        self.human_model = False
        self.AEBS_model = False
        # 0 broken; 1 broken+cost; 2 cost; 3 sparse; 4 sparse+cost
        self.reward_model = 0

        '''环境动态的活动变量'''
        self.curr_veh_id = None
        self.step_index = 0
        self.ped_list = []  # 记录当前交叉口车辆所观测行人
        self.curr_is_broken = False
        self.curr_is_finished = False
        self.curr_speed = 0
        self.curr_action = np.zeros(self.action_space.shape[0])
        self.start_step = 0
        '''agent-environment interaction framework variable'''
        self.observation = np.zeros(self.__veh_state_num + self.__MAX_PERSON_COUNT * self.__ped_state_num)
        self.reward = 0
        self.cost = 0
        self.risk = 0
        self.done = False
        self.info = {'sum_cost': 0}

    def start(self, label='naive', path="exp.sumocfg", gui=False, max_step=150, reward_model=2, is_human_model=False):
        self.path = path
        self.label = label
        self.gui = gui
        self.max_step = max_step
        self.reward_model = reward_model
        self.human_model = is_human_model
        if self.gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')
        cmd = [sumoBinary, "-c", self.path, ]
        traci.start(cmd, label=self.label)

    def calc_ttc(self, ped_id):
        assert self.curr_veh_id is not None
        delta_X = self.connect.vehicle.getPosition(self.curr_veh_id)[0] - self.connect.person.getPosition(ped_id)[0]
        if delta_X < 0:
            return 10
        ttc = delta_X / (self.curr_speed + 1E-10)
        ped_pred = self.connect.person.getPosition(ped_id)[1] + np.cos(
            np.radians(self.connect.person.getAngle(ped_id))) * self.connect.person.getSpeed(ped_id) * ttc
        if (ped_pred <= self.connect.vehicle.getPosition(self.curr_veh_id)[1] + 1.6) and (
                ped_pred >= self.connect.vehicle.getPosition(self.curr_veh_id)[1] - 1.6):
            return min(10, ttc)
        else:
            return 10

    def AEBS(self):
        # ttc = 10
        ttb = self.curr_speed / self.__decel
        # if len(self.ped_list) > 0:
        #     for ped_id in self.ped_list[0:self.__MAX_PERSON_COUNT]:
        #         ttc = np.min([self.calc_ttc(ped_id), ttc])
        # if np.min([ttb / ttc, 1.0])>0.9:
        START = False
        for ped_id in self.ped_list[0:self.__MAX_PERSON_COUNT]:
            a, ratio, ttc_1 = self.calc_cost(ped_id)
            if (ttb * ttc_1 >= 0.95) and (ratio >= 0.95):
                START = True
        if START:
            # if self.risk >= 0.5:
            self.__AEBS_state += 1
            self.curr_speed = max(self.connect.vehicle.getSpeed(
                self.curr_veh_id) - self.__AEBS_brake * self.__actionStepLength, 0)
            self.connect.vehicle.setSpeed(self.curr_veh_id, self.curr_speed)
            self.connect.vehicle.remove(self.curr_veh_id)
        #     if self.curr_speed == 0:
        #         self.connect.vehicle.remove(self.curr_veh_id)
        # else:
        #     if self.__AEBS_state > 0:
        #         if self.curr_speed > 0:
        #             self.__AEBS_state += 1
        #             self.curr_speed = max(self.connect.vehicle.getSpeed(
        #                 self.curr_veh_id) - self.__AEBS_brake * self.__actionStepLength, 0)
        #             self.connect.vehicle.setSpeed(self.curr_veh_id, self.curr_speed)
        #             if self.curr_speed == 0:
        #                 self.connect.vehicle.remove(self.curr_veh_id)

    def calc_risk(self, ped_id):
        assert self.curr_veh_id is not None
        veh_x, veh_y = self.connect.vehicle.getPosition(self.curr_veh_id)
        ped_x, ped_y = self.connect.person.getPosition(ped_id)
        veh_l, veh_w = self.connect.vehicle.getLength(self.curr_veh_id), self.connect.vehicle.getWidth(
            self.curr_veh_id) + 1
        if veh_x < ped_x:
            return 0
        Edge = []
        Edge.append([veh_x, veh_y + 1 / 2 * veh_w])
        Edge.append([veh_x, veh_y - 1 / 2 * veh_w])
        Edge.append([veh_x + veh_l, veh_y + 1 / 2 * veh_w])
        Edge.append([veh_x + veh_l, veh_y - 1 / 2 * veh_w])

        v_veh = self.connect.vehicle.getSpeed(self.curr_veh_id) + 1E-10
        d_ = (v_veh * v_veh) / (2 * self.__decel) + 20

        v_veh_max = v_veh + self.__accel * self.__actionStepLength * 0.8
        v_veh_min = v_veh - self.__decel * self.__actionStepLength * 0.8
        v_ped = self.connect.person.getSpeed(ped_id) * np.cos(
            np.radians(self.connect.person.getAngle(ped_id)))
        tan_theta_i = [(edge[1] - ped_y) / (edge[0] - ped_x) for edge in Edge]
        tan_theta_max = np.max(tan_theta_i)
        tan_theta_min = np.min(tan_theta_i)
        d = self.distance_vehicle_pedestrians(self.curr_veh_id, ped_id) + 1E-10
        v_sum = np.sqrt(v_veh * v_veh + v_ped * v_ped)
        tan_theta = v_ped / v_veh
        theta_v_2 = max(math.atan(v_ped / v_veh_max), math.atan(v_ped / v_veh_min))
        theta_v_1 = min(math.atan(v_ped / v_veh_max), math.atan(v_ped / v_veh_min))
        theta_max = math.atan(tan_theta_max)
        theta_min = math.atan(tan_theta_min)
        if not ((theta_v_1 >= theta_max) or (theta_v_2 <= theta_min)):
            ratio = (min(theta_max, theta_v_2) - max(theta_min, theta_v_1)) / (theta_v_2 - theta_v_1 + 1E-10)
        else:
            ratio = 0
        risk = v_sum / d * ratio
        return min(risk, 10)

    def calc_cost(self, ped_id):
        assert self.curr_veh_id is not None
        eta_1 = 4
        eta_2 = 10
        U_threshold = self.max_step / 10

        veh_x, veh_y = self.connect.vehicle.getPosition(self.curr_veh_id)
        ped_x, ped_y = self.connect.person.getPosition(ped_id)
        veh_l, veh_w = self.connect.vehicle.getLength(self.curr_veh_id), self.connect.vehicle.getWidth(
            self.curr_veh_id) + 1
        if veh_x < ped_x:
            return 0
        Edge = []
        Edge.append([veh_x, veh_y + 1 / 2 * veh_w])
        Edge.append([veh_x, veh_y - 1 / 2 * veh_w])
        Edge.append([veh_x + veh_l, veh_y + 1 / 2 * veh_w])
        Edge.append([veh_x + veh_l, veh_y - 1 / 2 * veh_w])

        v_veh = self.curr_speed + 1E-10
        d_ = (v_veh / 2) * (v_veh / self.__decel) + 10

        v_veh_max = v_veh + self.__accel * self.__actionStepLength * 0.8
        v_veh_min = v_veh - self.__decel * self.__actionStepLength * 0.8
        v_ped = self.connect.person.getSpeed(ped_id) * np.cos(
            np.radians(self.connect.person.getAngle(ped_id)))
        tan_theta_i = [(edge[1] - ped_y) / (edge[0] - ped_x) for edge in Edge]
        tan_theta_max = np.max(tan_theta_i)
        tan_theta_min = np.min(tan_theta_i)
        d = self.distance_vehicle_pedestrians(self.curr_veh_id, ped_id) + 1E-10
        v_sum = np.sqrt(v_veh * v_veh + v_ped * v_ped)
        tan_theta = v_ped / v_veh
        theta_v_2 = max(math.atan(v_ped / v_veh_max), math.atan(v_ped / v_veh_min))
        theta_v_1 = min(math.atan(v_ped / v_veh_max), math.atan(v_ped / v_veh_min))
        theta_max = math.atan(tan_theta_max)
        theta_min = math.atan(tan_theta_min)
        if not ((theta_v_1 >= theta_max) or (theta_v_2 <= theta_min)):
            ratio = (min(theta_max, theta_v_2) - max(theta_min, theta_v_1)) / (theta_v_2 - theta_v_1 + 1E-10)
        else:
            ratio = 0
        U_1, U_2 = 0, 1
        if d <= d_:
            U_1 = eta_1 * v_sum / d * ratio
            # U_2 = 1 / 2 * eta_2 * (1 / d - 1 / d_)
        # print('ped_id', person_id, 'tan_theta_max:', tan_theta_max, 'tan_theta_min:', tan_theta_min, 'tan_theta',
        #       tan_theta, 'ratio', ratio)
        return min(U_1 * U_2, U_threshold), ratio, v_sum / d

    def update_max_risk(self):
        self.risk = 0
        for ped_id in self.ped_list[0:self.__MAX_PERSON_COUNT]:
            self.risk = max(self.calc_cost(ped_id)[0], self.risk)
        return self.risk

    def update_reward(self):
        self.reward = 0
        self.info['collision_cost'] = self.cost = 0

        if (self.curr_veh_id is None):
            scale = 10
            '''碰撞奖励'''
            if (self.reward_model in [0, 1]):
                if self.curr_is_broken:
                    '''未来碰撞惩罚'''
                    self.info['collision_cost'] = self.cost = self.max_step / scale
                    self.reward = -(self.cost)
                    # return self.reward

            '''稀疏结束奖励'''
            if (self.reward_model in [3, 4]):
                if self.curr_is_finished:
                    # self.reward = 0
                    self.reward = self.max_step / (10 * scale)  # eposide 完成 奖励1.5
                else:  # eposide 未完成
                    if self.curr_is_broken:
                        '''未来碰撞惩罚'''
                        self.info['collision_cost'] = self.cost = self.max_step / scale
                        self.reward = -(self.cost-15)  # eposide发生碰撞 惩罚15
                    else:
                        # self.reward = -0
                        self.reward = -self.max_step / (10 * scale)  # eposide未发生碰撞但未完成 惩罚1.5
                # return self.reward

        '''速度奖励'''
        if self.curr_speed <= self.__MAX_SPEED:
            speed_reward = self.curr_speed / self.__MAX_SPEED
        else:
            speed_reward = -1

        '''急刹车惩罚'''
        brake_penanlty = 0
        self.info['aeb_cost'] = 0
        if self.__AEBS_state == 1:
            self.info['aeb_cost'] = 5
            brake_penanlty = -(self.info['aeb_cost'])

        '''保持车道中心的奖励'''
        if self.curr_veh_id is not None:
            lane_id = traci.vehicle.getLaneID(self.curr_veh_id)
            center_reward = - abs(self.connect.vehicle.getPosition(self.curr_veh_id)[1] - \
                                  self.connect.lane.getShape(lane_id)[0][1])
            # print(center_reward)
        else:
            center_reward = 0

        '''未来碰撞惩罚'''
        self.info['cost'] = self.info['collision_cost'] + self.info['aeb_cost']
        self.info['sum_cost'] += self.info['cost']

        if (self.reward_model in [0, 3]):
            self.reward = +(speed_reward + center_reward + brake_penanlty)
        else:
            self.reward = +(speed_reward + center_reward + brake_penanlty - self.risk)

        return self.reward

    def is_brake(self):
        veh_x_curr, veh_y = self.connect.vehicle.getPosition(self.curr_veh_id)
        radian = math.radians(self.connect.vehicle.getAngle(self.curr_veh_id))
        length = self.connect.vehicle.getLength(self.curr_veh_id) + 3  # 车前预留1.5米
        width = self.connect.vehicle.getWidth(self.curr_veh_id) + 1.4  # 车两边各预留0.5米
        move_distance = self.curr_speed * self.__actionStepLength
        center_point = clac_center_point(np.array([veh_x_curr, veh_y]), radian, move_distance, length)
        length = length + move_distance
        veh_obb = OBB(center_point, radian, length / 2, width / 2)

        for ped_id in self.ped_list:
            ped_x, ped_y = self.connect.person.getPosition(ped_id)
            radian = math.radians(self.connect.person.getAngle(ped_id))
            length = self.connect.person.getLength(ped_id)
            width = self.connect.person.getWidth(ped_id)
            move_distance = self.connect.person.getSpeed(ped_id) * self.__actionStepLength
            center_point = clac_center_point(np.array([ped_x, ped_y]), radian, move_distance, length)
            length = length + move_distance
            if isCollision(veh_obb, OBB(center_point, radian, length / 2, width / 2)):
                self.curr_is_broken = True
        return self.curr_is_broken

    def _update_most_closely_peds(self):
        self.ped_list = []
        if self.curr_veh_id is None:
            return self.ped_list
        veh_x, veh_y = self.connect.vehicle.getPosition(self.curr_veh_id)
        veh_rad = np.radians(self.connect.vehicle.getAngle(self.curr_veh_id))
        veh_v_vec = np.array([math.sin(veh_rad), math.cos(veh_rad)])
        for ped_id in self.connect.person.getIDList():
            ped_x, ped_y = self.connect.person.getPosition(ped_id)
            ped_rad = np.radians(self.connect.person.getAngle(ped_id))
            # 行人到车的向量
            veh_ped_vec = np.array([ped_x, ped_y]) - np.array([veh_x, veh_y])
            # 车身后的行人忽略
            if np.dot(veh_ped_vec, veh_v_vec) < 0:
                continue
            # 考虑行人朝向的车，行人身后的车忽略
            if np.dot(-veh_ped_vec, np.array([math.sin(ped_rad), math.cos(ped_rad)])) > 0:  # 向上的行人
                self.ped_list.append([ped_id, self.distance_vehicle_pedestrians(self.curr_veh_id, ped_id)])

        self.ped_list.sort(key=lambda person: person[1])
        self.ped_list = [self.ped_list[i][0] for i in range(len(self.ped_list))]
        return self.ped_list

    def update_obs(self):
        if self.curr_veh_id == None:
            # self.observation = np.zeros(self.__veh_state_num + self.__MAX_PERSON_COUNT * self.__ped_state_num)
            return self.observation

        self.observation = np.zeros(self.__veh_state_num + self.__MAX_PERSON_COUNT * self.__ped_state_num)
        self.observation[0:2] = self.connect.vehicle.getPosition(self.curr_veh_id)  # 车的位置 x y
        self.observation[2] = self.connect.vehicle.getSpeed(self.curr_veh_id)  # 车的速度 大小m/s
        self.observation[3] = np.cos(np.radians(self.connect.vehicle.getAngle(self.curr_veh_id)))  # 车的速度 方向cos值，上为0度顺时针
        self.observation[4] = self.connect.vehicle.getLength(self.curr_veh_id)  # 车的长度
        self.observation[5] = self.connect.vehicle.getWidth(self.curr_veh_id)  # 车的宽度

        ped_index = 0  # 行人
        for ped_id in self.ped_list[0:self.__MAX_PERSON_COUNT]:
            obs_index = self.__veh_state_num + ped_index * self.__ped_state_num
            self.observation[obs_index:obs_index + 2] = self.connect.person.getPosition(ped_id)  # 行人的位置 x y
            self.observation[obs_index + 2] = self.connect.person.getSpeed(ped_id)  # 行人的速度 大小m/s
            self.observation[obs_index + 3] = np.cos(
                np.radians(self.connect.person.getAngle(ped_id)))  # 行人的速度 方向cos值，上为0度顺时针
            self.observation[obs_index + 4] = self.calc_ttc(ped_id)  # 车与行人的碰撞风险
            ped_index += 1

        return self.observation

    # throttle and brake
    def set_action(self, action=None):
        if action.any() == None:
            action = np.zeros(self.action_space.shape[0])
        assert type(action) is np.ndarray

        if action[0] < 0:
            self.curr_action[0] = max(self.action_space.low[0], action[0]) * self.__decel
        else:
            self.curr_action[0] = min(self.action_space.high[0], action[0]) * self.__accel
        self.curr_speed = max(self.connect.vehicle.getSpeed(
            self.curr_veh_id) + self.curr_action[0] * self.__actionStepLength, 0)
        self.connect.vehicle.setSpeed(self.curr_veh_id, self.curr_speed)

    # throttle and brake,steering
    # def set_action(self, action=None):
    #     if action.any() == None:
    #         action = np.zeros(2)
    #     assert type(action) is np.ndarray
    #
    #     if action[0] < 0:
    #         self.curr_action[0] = max(self.action_space.low[0], action[0]) * self.__decel
    #     else:
    #         self.curr_action[0] = min(self.action_space.high[0], action[0]) * self.__accel
    #     self.curr_speed = max(self.connect.vehicle.getSpeed(
    #         self.curr_veh_id) + self.curr_action[0] * self.__actionStepLength, 0)
    #     self.connect.vehicle.setSpeed(self.curr_veh_id, self.curr_speed)
    #     self.curr_action[1] = action[1]
    #     self.connect.vehicle.changeSublane(self.curr_veh_id, self.curr_action[1])

    def step(self, action=None):
        '''env_dynamic'''
        if not self.human_model:
            self.set_action(action)
        else:
            self.curr_speed = self.connect.vehicle.getSpeed(self.curr_veh_id)
        self.update_max_risk()
        if self.AEBS_model:
            self.AEBS()
        # if  self.is_brake_standard():

        # else:
        #     if self.curr_speed == 0:  # 速度为0将结束
        #         self.connect.vehicle.remove(self.curr_veh_id)
        if not self.__AEBS_state:
            if self.is_brake():  # 发生碰撞将结束
                self.connect.vehicle.remove(self.curr_veh_id)

        self.connect.simulationStep()
        self.step_index += 1

        # (len(self.connect.vehicle.getIDList()) == 0) 包含 完成任务 或 self.curr_is_broken==Ture 或 self.curr_speed == 0
        if (len(self.connect.vehicle.getIDList()) == 0) or (self.step_index >= self.max_step):
            self.curr_veh_id = None
            # if (self.step_index < self.max_step) and (not self.curr_speed == 0) and (not self.curr_is_broken):
            if (self.step_index < self.max_step) and (not self.curr_is_broken) and (self.__AEBS_state == 0):
                self.curr_is_finished = True
        self._update_most_closely_peds()

        '''收集数据'''
        if self.curr_veh_id == None:
            self.done = True
        done = self.done
        observation = self.update_obs()
        reward = self.update_reward()
        info = self.evaluation()

        return observation, reward, done, info

    def reset(self):
        '''reset env'''
        self.connect = traci.getConnection(self.label)
        self.connect.load(["-c", self.path])
        '''init curr_veh_id'''
        while True:
            self.connect.simulationStep()
            if len(self.connect.vehicle.getIDList()) > 0:
                self.curr_veh_id = self.connect.vehicle.getIDList()[0]
                break
        '''init 环境动态的活动变量'''
        self.step_index = 0
        self.done = False
        self.info = {'sum_cost': 0}
        self._init_veh()  # 初始化 veh变量
        self._update_most_closely_peds()
        return self.update_obs()

    def set_startstep(self, start_step):
        self.start_step = start_step

    # 在reset()中,重置车辆后，执行
    def _init_veh(self, p_p=160, v_sigma=5, v_mu=2):
        if self.human_model:
            v_sigma = 13
            v_mu = 1
        self.__actionStepLength = self.connect.vehicle.getActionStepLength(self.curr_veh_id)
        self.__MAX_SPEED = self.connect.lane.getMaxSpeed(self.connect.vehicle.getLaneID(self.curr_veh_id))
        self.__AEBS_state = 0
        self.connect.vehicle.setSpeedMode(self.curr_veh_id, sm=8)
        self.connect.vehicle.setLaneChangeMode(self.curr_veh_id, lcm=2218)
        self.curr_is_broken = False
        self.curr_is_finished = False

        self.connect.vehicle.setSpeed(self.curr_veh_id, 0)
        self.start_step = np.random.randint(0, p_p, 1)[0]
        for i in range(self.start_step):
            self.connect.simulationStep()
        '''initail veh lane'''
        edge_id = self.connect.vehicle.getRoadID(self.curr_veh_id)
        x, y = self.connect.vehicle.getPosition(self.curr_veh_id)
        angle = self.connect.vehicle.getAngle(self.curr_veh_id)
        # lane_id=np.random.randint(0, traci.edge.getLaneNumber(edge_id), 1)[0]
        lane_id = 1
        self.connect.vehicle.moveToXY(self.curr_veh_id, edge_id, lane_id, x, y, angle)
        '''initial veh speed'''
        self.curr_speed = np.clip(np.random.normal(v_sigma, v_mu, 1), v_sigma - 1.96 * v_mu, v_sigma + 1.96 * v_mu)[0]
        self.connect.vehicle.setSpeed(self.curr_veh_id, self.curr_speed)
        self.curr_action = np.zeros(self.action_space.shape[0])
        self.connect.simulationStep()
        if self.human_model:
            self.connect.vehicle.setSpeedMode(self.curr_veh_id, sm=31)
        self.connect.simulationStep()

    def evaluation(self):
        if self.curr_veh_id is None:
            self.info['speed'] = self.curr_speed
            self.info['crash'] = self.curr_is_broken
            self.info['success'] = self.curr_is_finished
            self.info['AEB_num'] = (self.__AEBS_state == 1)
            self.info['fuel'] = 0
            return self.info

        self.info['veh_id'] = self.curr_veh_id
        self.info['speed'] = self.curr_speed
        self.info['crash'] = self.curr_is_broken
        self.info['success'] = self.curr_is_finished
        self.info['AEB_num'] = (self.__AEBS_state == 1)
        self.info['fuel'] = self.connect.vehicle.getFuelConsumption(
            self.curr_veh_id) * self.__actionStepLength
        return self.info

    def render(self, mode='human'):
        return None

    def close(self):
        traci.close()

    def distance_vehicle_pedestrians(self, veh_id, ped_id):
        position_veh = np.array(self.connect.vehicle.getPosition(veh_id))
        position_ped = np.array(self.connect.person.getPosition(ped_id))
        return np.linalg.norm(position_veh - position_ped)

    def set_label(self, label):
        self.label = label


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(rootPath + "/test")

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    env = gym.make("sumo_env-v6")
    env.start(gui=False, path="/home/user/repos/CAV/envs/multi_lane_v1/exp.sumocfg")
    env.max_step = 1000
    car_number = 100
    action = np.array([0.4])
    collision_rate = []
    success_rate = []
    for iter in range(100):
        crash_num = 0
        success_num = 0
        print('iter ', iter)
        for i in range(car_number):
            observation = env.reset()
            step = 0
            total_reward = 0
            total_cost = 0
            done = False
            a = False
            while not done:
                observation, reward, done, info = env.step(action)

                '''random policy'''
                # action = env.action_space.sample()
                '''policy1'''
                if info['cost'] > 0:
                    action[0] = -1
                    a = True
                else:
                    # if env.curr_speed == 0:
                    #     a = False
                    if a:
                        action[0] = 0
                    else:
                        if env.curr_speed <= 15:
                            action[0] = 1
                        else:
                            action[0] = -1
                    a = False
                total_reward += env.reward
                total_cost += env.cost
                # print(env.reward, env.cost, env.curr_speed, action)
                step += 1
                # time.sleep(0.05)
            crash_num += info['crash']
            success_num += info['success']
            # print('Finished', i, 'Length', step, 'Crash', info['crash'], 'Finished', info['success'], 'R:',
            #       total_reward,
            #       'C:', total_cost)
            # print()
        print('crash', crash_num)
        print('success', success_num)
        print('done', car_number)
        writer.add_scalar('human/CollisionRate', crash_num / car_number, iter)
        writer.add_scalar('human/SuccessRate', success_num / car_number, iter)
        collision_rate.append(crash_num / car_number)
        success_rate.append(success_num / car_number)

    print('crash', crash_num)
    print('success', success_num)
    print('done', car_number)
    print(collision_rate)
    print(np.mean(collision_rate))
    print(success_rate)
    print(np.mean(success_rate))
