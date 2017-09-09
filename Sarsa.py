from Easy21 import Easy21
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

class Sarsa(object):
    def __init__(self, lambda_value):
        self.q_value = np.zeros((10, 21, 2))
        self.count = np.zeros((10, 21, 2))
        self.eligibility_trace = np.zeros((10, 21, 2))
        self.constant = 100.0
        self.lambda_value = lambda_value
        self.env = Easy21()

    def epsilon_greedy_action(self, state):
        epsilon = self.constant / (self.constant + np.sum(self.count[state[0]-1, state[1]-1, :]))
        if random.random() < epsilon:
            return np.random.randint(2)
        else:
            return np.argmax(self.q_value[state[0]-1, state[1]-1, :])

    def update_q_value(self, state, action, next_state, reward, next_action):
        self.eligibility_trace[state[0]-1, state[1]-1, action] += 1
        if next_action:
            td_error = reward + self.q_value[next_state[0]-1, next_state[1]-1, next_action] - \
                    self.q_value[state[0]-1, state[1]-1, action]
        else:
            td_error = reward - self.q_value[state[0]-1, state[1]-1, action]
        alpha = 1.0 / self.count[state[0]-1, state[1]-1, action]
        self.q_value += alpha*td_error*self.eligibility_trace
        self.eligibility_trace *= self.lambda_value

    def train(self, episode_number=1000, q_value=None, is_mse=False):
        mses = []
        win = 0
        for i in range(episode_number):
            self.eligibility_trace = np.zeros((10, 21, 2))
            state = self.env.reset()
            action = self.epsilon_greedy_action(state)
            self.count[state[0] - 1, state[1] - 1, action] += 1
            while True:
                next_state, reward, done, _ = self.env.step(action)
                if not done:
                    next_action = self.epsilon_greedy_action(next_state)
                else:
                    next_action = None
                self.update_q_value(state, action, next_state, reward, next_action)
                if done:
                    if reward > 0:
                        win += 1
                    if is_mse:
                        mse = np.mean((q_value - self.q_value) ** 2)
                        mses.append(mse)
                    if (i+1) % 1000 == 0:
                        print "win", win, "episode", (i+1)
                    break
                state = next_state
                action = next_action
                self.count[state[0] - 1, state[1] - 1, action] += 1
        return mses

if __name__ == '__main__':
    f = open("q_value.pkl", 'r')
    q_optimal_value = pickle.load(f)

    lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_mse = []
    for l in lambdas:
        print "lambda", l
        td = Sarsa(l)
        td.train()
        mse = np.mean((q_optimal_value - td.q_value)**2)
        all_mse.append(mse)
    fig = plt.figure()
    fig.suptitle("Sarsa")
    ax = fig.add_subplot(111)
    ax.plot(lambdas, all_mse)
    ax.set_xlabel("lambda")
    ax.set_ylabel("Mean Square Error")
    plt.show()

    # td =Sarsa(0)
    # all_mse_1 = td.train(episode_number=10000, q_value=q_optimal_value, is_mse=True)
    # td = Sarsa(1.0)
    # all_mse_2 = td.train(episode_number=10000, q_value=q_optimal_value, is_mse=True)
    # fig = plt.figure()
    # fig.suptitle("Sarsa")
    # ax = fig.add_subplot(111)
    # ax.plot(range(1, len(all_mse_1)+1), all_mse_1, label="lambda=0")
    # ax.plot(range(1, len(all_mse_2) + 1), all_mse_2, label="lambda=1")
    # ax.set_xlabel("episode number")
    # ax.set_ylabel("Mean Square Error")
    # ax.legend()
    # plt.show()

