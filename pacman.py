import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from nes_py.wrappers import JoypadSpace
import gym
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np

env = gym.make('Contra-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

print("actions", env.action_space)
print("actions", env.action_space.n)
print("observation_space ", env.observation_space.shape[0])

from FaceGameController import FaceGameController

controller = FaceGameController(debug=True,use_gpu=True)


def contoact(control):
    '''
                                    a               b
    ["left","right","up","down","mopen","mheri","lefteye","righteye"]
    [1,        2,     4,    8,    16,      0,       64,       128]
    '''
    if control == 0:
        return 0
    if control == 2:
        return 1
    if control == 18:
        return 2
    if control == 66:
        return 3
    if control == 22:
        return 4
    if control == 70:
        return 5
    if control == 86:
        return 6
    if control == 16:
        return 7
    if control == 64:
        return 8
    if control == 80:
        return 9
    if control == 1:
        return 10
    if control == 17:
        return 11
    if control == 65:
        return 12
    if control == 21:
        return 13
    if control == 69:
        return 14
    if control == 85:
        return 15
    if control == 24:
        return 16
    if control == 72:
        return 17
    if control == 88:
        return 18
    if control == 20:
        return 19
    if control == 84:
        return 20
    else:
        return 0

input()
done = False
env.reset()
i = 0 
while True:
    if done:
        print("Over")
        break
    res = controller.control(use_gpu=True)
    res = contoact(np.sum(res))
    state, reward, done, info = env.step(res)
    env.render()
    #input()

env.close()