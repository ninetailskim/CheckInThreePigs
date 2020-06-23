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

import cv2


game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)

actionset = p.getActionSet()

print(actionset)
print(len(actionset))



p.init()
reward = 0.0

nb_frames = 10000

for i in range(nb_frames):
    if p.game_over():
        p.reset_game()

    observation = p.getScreenRGB()
    print(observation.shape)
    #action = agent.pickAction(reward, observation)
    reward = p.act(119)
    observation = cv2.transpose(observation)
    font = cv2.FONT_HERSHEY_SIMPLEX
    observation = cv2.putText(observation, '1', (0, 25), font, 1.2, (255, 255, 255), 2)
    cv2.imshow("ss", observation)
    cv2.waitKey(10)