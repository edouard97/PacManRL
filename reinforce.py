import itertools
import random
import numpy as np


# Uses algorithm REINFORCE with 2 sigmoids reprensenting the policy
# Poor results : not able to get small balls even when no ghost

def sigmoid(theta, x):
    assert len(theta) == len(x)
    return 1 / (1 + np.exp(-sum([theta[i]*x[i] for i in range(len(theta))])))


def apply_policy(theta1, theta2, s):
    a = random.random() <= sigmoid(theta1, s)
    b = random.random() <= sigmoid(theta2, s)
    if a and b:
        return 0
    elif a and not b:
        return 1
    elif b and not a:
        return 2
    else:
        return 3


def policy_gradient(states, actions, rewards, theta):
    assert len(states) == len(actions) == len(rewards)
    n = len(states)
    state_dim = len(states[0])
    gradient = [0]*state_dim
    cur_reward = sum(rewards)
    for i in range(n):
        factor = cur_reward
        if actions[i]:
            factor *= 1-sigmoid(theta, states[i])
        else:
            factor *= -sigmoid(theta, states[i])
        for j in range(state_dim):
            gradient[j] += factor*states[i][j]
        cur_reward -= rewards[i]
    return gradient


def update_theta(theta, alpha, states, actions, rewards):
    state_dim = len(states[0])
    gradient = policy_gradient(states, actions, rewards, theta)
    return [theta[i] + alpha * gradient[i] for i in range(state_dim)]


def update_alpha(start_alpha, i):
    return start_alpha


def vectorize(observation):
    ghost_positions = list(itertools.chain.from_iterable(observation[2]))
    return list(observation[0].flatten()) + list(observation[1]) + ghost_positions


def devectorize(state, map_shape):
    map_ = np.array(state[:map_shape[0]*map_shape[1]]).reshape(map_shape)
    ghosts = state[map_shape[0]*map_shape[1]:-2]
    pac_man = state[-2:]
    return map_, ghosts, pac_man
