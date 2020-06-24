import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

from ple.games.flappybird import FlappyBird
from ple import PLE
import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
from datetime import datetime
import cv2

LEARNING_RATE = 1e-3

class Model(parl.Model):
    def __init__(self, act_dim):
        self.fc0 = layers.fc(size=64, act='tanh', name="pgfc0")
        self.fc1 = layers.fc(size=32, act='tanh', name="pgfc1")
        self.fc2 = layers.fc(size=16, act='tanh', name="pgfc2")
        self.fc3 = layers.fc(size=act_dim, act='softmax', name="pgfc3")

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        out = self.fc0(obs)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr=None):
        """ Policy Gradient algorithm
        
        Args:
            model (parl.Model): policy的前向网络.
            lr (float): 学习率.
        """

        self.model = model
        assert isinstance(lr, float)
        self.lr = lr

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        return self.model(obs)

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        act_prob = self.model(obs)  # 获取输出动作概率
        # log_prob = layers.cross_entropy(act_prob, action) # 交叉熵
        log_prob = layers.reduce_sum(
            -1.0 * layers.log(act_prob) * layers.one_hot(
                action, act_prob.shape[1]),
            dim=1)
        cost = log_prob * reward
        cost = layers.reduce_mean(cost)

        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost

class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim,e_greed=0.1, e_greed_decrement=0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):  # 搭建计算图用于 更新policy网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim) 
        else:
            obs = np.expand_dims(obs, axis=0)  # 增加一维维度
            act_prob = self.fluid_executor.run(
                self.pred_program,
                feed={'obs': obs.astype('float32')},
                fetch_list=[self.act_prob])[0]
            act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
            #print("sample", act_prob)
            act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        print("predict", act_prob)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost

import random
import collections
import numpy as np


totale = 0
def run_episode(env, agent):
    actionset = env.getActionSet()
    obs_list, action_list, reward_list = [], [], []
    env.init()
    env.reset_game()
    
    while True:
        obs = list(env.getGameState().values())
        obs_list.append(obs)

        action = agent.sample(obs) # 采样动作
        #print(action)
        #input()
        action_list.append(action)

        reward = env.act(actionset[action])
        done = env.game_over()
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


def evaluate(env, agent, render=False):
    env = PLE(game, fps=30, display_screen=True)
    actionset = env.getActionSet()
    eval_reward = []
    for i in range(5):
        env.init()
        env.reset_game()
        obs = list(env.getGameState().values())
        episode_reward = 0
        while True:
            action = agent.predict(obs) # 选取最优动作
            # print(action)
            observation = env.getScreenRGB()
            score  = env.score()
            #action = agent.pickAction(reward, observation)
            observation = cv2.transpose(observation)
            font = cv2.FONT_HERSHEY_SIMPLEX
            observation = cv2.putText(observation, str(int(score)), (0, 25), font, 1.2, (255, 255, 255), 2)
            cv2.imshow("ss", observation)
            cv2.waitKey(10)  # 预测动作，只选最优动作
            reward= env.act(actionset[action])
            obs = list(env.getGameState().values())
            isOver = env.game_over()
            episode_reward += reward

            if isOver:
                break
        eval_reward.append(episode_reward)
        cv2.destroyAllWindows()
    return np.mean(eval_reward)


def calc_reward_to_go(reward_list, gamma=1.0):
    gamma_d = gamma / 10
    # gamma = 0
    # reward_list[-1] = -10
    for i in range(len(reward_list) - 3, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        if reward_list[i + 1] > reward_list[i] * 1.15:
            reward_list[i] += 0.5 * reward_list[i + 1]

    for i in range(len(reward_list) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_list[i] += gamma * reward_list[-1]
          # Gt
        gamma = max(0, gamma - gamma_d)
    return np.array(reward_list)

game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)  # CartPole-v0: 预期最后一次评估总分 > 180（最大值是200）
action_dim = len(env.getActionSet())  # CartPole-v0: 2
obs_shape = len(env.getGameState())  # CartPole-v0: (4,)

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.9, e_greed_decrement=1e-6)

# 加载模型
# save_path = './dqn_model.ckpt'
# agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够

max_episode = 20000000

# 开始训练
episode = 0

ps = datetime.now()

evmax = 0
i = 0

for i in range(max_episode):
    obs_list, action_list, reward_list = run_episode(env, agent)
    if i % 10 == 0:
        logger.info("Episode {}, Reward Sum {}.".format(
            i, sum(reward_list)))

    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)
    # print(reward_list)
    batch_reward = calc_reward_to_go(reward_list)
    # print(batch_reward)
    # input()
    agent.learn(batch_obs, batch_action, batch_reward)
    if (i + 1) % 100 == 0:
        eval_reward = evaluate(env, agent, render=False) # render=True 查看渲染效果，需要在本地运行，AIStudio无法显示
        logger.info('iter: {}  greedy: {} Test reward: {}'.format(i,agent.e_greed,eval_reward))
# 训练结束，保存模型
        if eval_reward > evmax:
            save_path = './modelpg2_' + str(episode) + '_' + str(eval_reward) + '.ckpt'
            agent.save(save_path)
            evmax = eval_reward