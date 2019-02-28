import itertools
import random
import numpy as np
import queue



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
    return start_alpha /np.sqrt(1+i)


def voisins(position, observation, game):
    i = position[0]
    j = position[1]
    n = len(observation[0])
    neighbours = []
    
    if i<n-1 and game[i+1][j] != 1:
        neighbours.append(((i+1, j), 'down'))
    if i>0 and game[i-1][j] != 1:
        neighbours.append(((i-1, j), 'up'))
    if j<n-1 and game[i][j+1] != 1:
        neighbours.append(((i, j+1), 'right'))
    if j>0 and game[i][j-1] != 1:
        neighbours.append(((i, j-1), 'left'))
        
    return neighbours
  
def retrace(visited, pos):
    last_move = visited[pos]
    if last_move:
        if last_move == 'up':
            pos = (pos[0]+1, pos[1])
            return retrace(visited, pos)+['up']
        if last_move == 'down':
            pos = (pos[0]-1, pos[1])
            return retrace(visited, pos)+['down']      
        if last_move == 'right':
            pos = (pos[0], pos[1]-1)
            return retrace(visited, pos)+['right']     
        if last_move == 'left':
            pos = (pos[0], pos[1]+1)
            return retrace(visited, pos)+['left']
    return []

def transform(observation):
    pac_position = observation[1]
    ghost_positions = observation[2]
    game = observation[0]
    
    q = queue.Queue()
    q.put(pac_position)
    visited = {}
    visited[pac_position] = None
    
    while not q.empty():
        current = q.get()

        if game[current[0]][current[1]] == 2: #small ball
            a =  retrace(visited, current)
            print(a)
            return a
        
        neighbours = voisins(current, observation, game)
        
        for nei in neighbours:
            if not nei[0] in visited:
                q.put(nei[0])
                visited[nei[0]]= nei[1]




def vectorize(observation):
    if len(observation) == 3:
        ghost_positions = list(itertools.chain.from_iterable(observation[2]))
        return list(observation[0].flatten()) + list(observation[1]) + ghost_positions
    else:
        ghost_positions = list(itertools.chain.from_iterable(observation[1]))
        return list(observation[0]) + ghost_positions
