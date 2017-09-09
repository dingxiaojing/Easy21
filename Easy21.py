from gym import spaces
import numpy as np
import random


class Easy21(object):
    def __init__(self):
        # 2 actions: 0-hits, 1-sticks
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete([[1, 10], [1, 21]])
        self.state = None

    def reset(self):
        player_card = np.random.randint(1, 11)
        dealer_card = np.random.randint(1, 11)
        self.state = (dealer_card, player_card)
        return np.array(self.state)

    def get_card(self):
        card = np.random.randint(1, 11)
        if random.random() < 2.0/3:
            return card
        else:
            return -card

    def step(self, action):
        dealer_card_up, player_sum = self.state
        if action == 0:
            player_sum += self.get_card()
            self.state = (dealer_card_up, player_sum)
            if player_sum > 21 or player_sum < 1:
                return self.state, -1.0, True, None
            return self.state, 0, False, None
        if action == 1:
            dealer_sum = dealer_card_up
            while True:
                dealer_sum += self.get_card()
                if dealer_sum > 21 or dealer_sum < 1:
                    return self.state, 1.0, True, None
                if dealer_sum >= 17:
                    if dealer_sum > player_sum:
                        return self.state, -1.0, True, None
                    elif dealer_sum < player_sum:
                        return self.state, 1.0, True, None
                    else:
                        return self.state, 0, True, None


if __name__ == '__main__':
    env = Easy21()
    win = 0
    for i in range(100):
        state = env.reset()
        while True:
            state, reward, done, _ = env.step(np.random.randint(2))
            if done:
                if reward > 0:
                    win += 1
                break
    print "win", win

