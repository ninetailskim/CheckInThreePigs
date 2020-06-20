import os
import gym
import numpy as np

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger
import random
import copy
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear


LEARNING_RATE = 0.005

class Model(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hid1_size = act_dim * 10
        
        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.conv1 = layers.conv2d(num_filters=2, filter_size=3, act='relu')
        self.fc2 = layers.fc(size=act_dim, act='softmax')
        self.conv2 = layers.conv2d(num_filters=2, stride=2, filter_size=3, act='relu')
        self.conv3 = layers.conv2d(num_filters=4, stride=2, filter_size=3, act='relu')
        self.conv4 = layers.conv2d(num_filters=8, stride=2, filter_size=3, act='relu')
        

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        #out = self.conv1(obs)
        out = self.conv2(obs)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


from parl.algorithms import PolicyGradient


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[1, 80, 80], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):  # 搭建计算图用于 更新policy网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[1, 80, 80], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # 增加一维维度
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
        act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        # print(act.shape)
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        # print(obs.shape)
        # print(act.shape)
        # print(reward.shape)
        # input()
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost


def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    last_obs = np.zeros((1,80,80))
    while True:

        obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
        #obs_list.append(np.concatenate((obs,last_obs), axis=0))
        #last_obs = copy.deepcopy(obs)
        obs_list.append(obs + last_obs * 0.5)
        last_obs=obs
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    #print("Finish")
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        last_obs = np.zeros((1,80,80))
        while True:
            obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs + last_obs*0.5) # 选取最优动作
            last_obs = obs
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)



def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195] # 裁剪
    image = image[::2,::2,0] # 下采样，缩放2倍
    image[image == 144] = 0 # 擦除背景 (background type 1)
    image[image == 109] = 0 # 擦除背景 (background type 2)
    image[image != 0] = 1 # 转为灰度图，除了黑色外其他都是白色
    #return image.astype(np.float).ravel()
    #np.expand_dims(obs, axis=0)
    return np.expand_dims(image.astype(np.float), axis=0)


# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


# 创建环境
env = gym.make('Pong-v0')
obs_dim = 80 * 80
act_dim = env.action_space.n
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

model = Model(act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)


for i in range(1000):
    obs_list, action_list, reward_list = run_episode(env, agent)
    # if i % 10 == 0:
    #     logger.info("Train Episode {}, Reward Sum {}.".format(i, 
    #                                         sum(reward_list)))

    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)
    batch_reward = calc_reward_to_go(reward_list)

    agent.learn(batch_obs, batch_action, batch_reward)
    if i < 300:
        if i % 30 == 0:
            total_reward = evaluate(env, agent, render=False)
            logger.info('Episode {}, Test reward: {}'.format(i, 
                                            total_reward))
    else:
        if i % 100 == 0:
            total_reward = evaluate(env, agent, render=False)
            logger.info('Episode {}, Test reward: {}'.format(i, 
                                            total_reward))
# save the parameters to ./model.ckpt
    if i % 100 == 0:    
        agent.save('./modelconv2_'+ str(LEARNING_RATE) + '_' + str(i) +'.ckpt')