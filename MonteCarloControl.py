from Easy21 import Easy21
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import pickle
from mpl_toolkits.mplot3d import Axes3D


class MonteCarloControl(object):
    def __init__(self):
        self.q_value = np.zeros((10, 21, 2))
        self.count = np.zeros((10, 21, 2))
        self.constant = 100.0
        self.env = Easy21()

    def epsilon_greedy_action(self, state):
        dealer_card, player_sum = state
        dealer_card -= 1
        player_sum -= 1
        state_count = np.sum(self.count[dealer_card, player_sum, :])
        epsilon = self.constant / (self.constant + state_count)
        if random.random() < epsilon:
            return np.random.randint(2)
        else:
            return np.argmax(self.q_value[dealer_card, player_sum, :])

    def update_q_value(self, state, action, reward):
        dealer_card, player_sum = state
        dealer_card -= 1
        player_sum -= 1
        alpha = 1.0 / self.count[dealer_card, player_sum, action]
        self.q_value[dealer_card, player_sum, action] += alpha * (reward - self.q_value[dealer_card, player_sum, action])

    def train(self, iteration=10000):
        count_win = 0
        for i in range(iteration):
            trajectory = []
            total_reward = 0
            state = self.env.reset()
            action = self.epsilon_greedy_action(state)
            trajectory.append((state, action))
            self.count[state[0] - 1, state[1] - 1, action] += 1
            while True:
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    if reward > 0:
                        count_win += 1
                    if (i+1) % 10000 == 0:
                        print "episode", (i+1), "win", count_win
                    for t in trajectory:
                        self.update_q_value(t[0], t[1], total_reward)
                    break
                action = self.epsilon_greedy_action(state)
                trajectory.append((state, action))
                self.count[state[0] - 1, state[1] - 1, action] += 1

    def plot_state_value(self):
        fig = plt.figure()
        fig.suptitle("State Value")
        ax = fig.add_subplot(111, projection='3d')
        X = range(1, 11)
        Y = range(1, 22)
        Z = np.max(self.q_value, axis=2)
        X, Y = np.meshgrid(Y, X)
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('player sum')
        ax.set_ylabel('dealer showing')
        ax.set_zlabel('reward')
        plt.show()

if __name__ == '__main__':
    mc = MonteCarloControl()
    mc.train(1000000)
    f = open('q_value.pkl', 'w')
    pickle.dump(mc.q_value, f)
    mc.plot_state_value()
