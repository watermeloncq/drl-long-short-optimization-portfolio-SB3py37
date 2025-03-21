import gym.wrappers
import numpy as np
from ..util import tanh


class TanhActions(gym.Wrapper):
    """
    Environment wrapper to tanh actions.

    Usage:
        env = gym.make('Pong-v0')
        env = TanhActions(env)

    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md

    """

    def step(self, action):
        # also it puts it in a list
        if isinstance(action, list):
            action = action[0]

        if isinstance(action, dict):
            action = list(action[k] for k in sorted(action.keys()))

        action = tanh(action, t=1)
        action[0] = np.clip(action[0], 0, 1)
        action[1:] = np.clip(action[1:], -0.96, 0.96)

        # # arbitrage setting
        # n = np.size(action)
        # x1 = x2 = 0
        # for i in range(1,n):
        #     x1 += action[i]
        #     x2 += np.abs(action[i])

        # if x1 == x2 and x2-np.abs(action[-1]) != 0:
        #     action[-1] = -action[-1]

        # scale actions
        # action_abs = np.abs(action)
        # action = action / action_abs.sum()

        return self.env.step(action)
