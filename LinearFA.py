from Easy21 import Easy21
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random


class LinearFA(object):
    def __init__(self, lambda_value):
        self.theta = np.zeros(36)
        self.epsilon = 0.05
        self.alpha = 0.01
        self.eligibility_trace = np.zeros(36)
        self.dealer_feature_range = [[1, 4], [4, 7], [7, 10]]
        self.player_feature_range = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
        self.lambda_value = lambda_value
        self.env = Easy21()

    def feature(self, state, action):
        feat = []
        for i in range(3):
            for j in range(6):
                for k in range(2):
                    d = self.dealer_feature_range[i]
                    p = self.player_feature_range[j]
                    if d[0] <= state[0] <= d[1] and p[0] <= state[1] <= p[1] and action == k:
                        feat.append(1)
                    else:
                        feat.append(0)
        return np.array(feat)

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(2)
        else:
            q_value_hit = np.sum(self.theta * self.feature(state, 0))
            q_value_stick = np.sum(self.theta * self.feature(state, 1))
            if q_value_hit > q_value_stick:
                return 0
            else:
                return 1

    def update(self, state, action, next_state, reward, next_action):
        feat = self.feature(state, action)
        next_feat = self.feature(next_state, next_action)
        self.eligibility_trace += feat
        if next_action:
            td_error = reward + np.sum(self.theta * next_feat) - np.sum(self.theta * feat)
        else:
            td_error = reward - np.sum(self.theta * feat)
        self.theta += self.alpha * td_error * self.eligibility_trace
        self.eligibility_trace *= self.lambda_value

    def get_q_value(self):
        q_value = np.zeros((10, 21, 2))
        for i in range(1, 11):
            for j in range(1, 22):
                for k in range(2):
                    feat = self.feature((i, j), k)
                    q_value[i-1, j-1, k] = np.sum(feat * self.theta)
        return q_value

    def train(self, episode_number=1000, mc_q_value=None, is_mse=False):
        mses = []
        win = 0
        for i in range(episode_number):
            self.eligibility_trace = np.zeros(36)
            state = self.env.reset()
            action = self.epsilon_greedy_action(state)
            while True:
                next_state, reward, done, _ = self.env.step(action)
                if not done:
                    next_action = self.epsilon_greedy_action(next_state)
                else:
                    next_action = None
                self.update(state, action, next_state, reward, next_action)
                if done:
                    if reward > 0:
                        win += 1
                    if is_mse:
                        mse = np.mean(np.power(self.get_q_value() - mc_q_value, 2))
                        mses.append(mse)
                    if (i + 1) % 1000 == 0:
                        print "win", win, "episode", (i+1)
                    break
                state = next_state
                action = next_action
        return mses

if __name__ == "__main__":
    f = open("q_value.pkl", 'r')
    q_optimal_value = pickle.load(f)
    fa = LinearFA(0)
    all_mse_1 = fa.train(episode_number=1000, mc_q_value=q_optimal_value, is_mse=True)
    fa = LinearFA(1.0)
    all_mse_2 = fa.train(episode_number=1000, mc_q_value=q_optimal_value, is_mse=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(all_mse_1)+1), all_mse_1, label="lambda=0")
    ax.plot(range(1, len(all_mse_2) + 1), all_mse_2, label="lambda=1")
    ax.legend()
    ax.subtitle("Linear Function Approximation")
    ax.set_xlabel("episode number")
    ax.set_ylabel("Mean Square Error")
    plt.show()
