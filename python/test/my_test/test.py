
import tensorflow as tf
import random
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import random

class random_producers():

  class producer():
    def __init__(self):
      self.loc = np.random.uniform(0,10)
      self.scale = 1
      self.random_prob = partial(np.add,self.loc,0)#partial(np.random.normal,self.loc,self.scale)
    def __call__(self):
      return self.random_prob()

  def __init__(self,n):
    self.n = n
    self.producers = [self.producer() for i in range(n)]
    self.max_id = max((self.producers[i].loc for i in range(self.n)))



  def get_random(self):
    return np.array([self.producers[i]() for i in range(self.n)])

  @property
  def expected_max_reward_id(self):
    return self.max_id

class n_armed_bandit():
  def __init__(self, n, _random_producers=None):
    self.n = n
    if _random_producers == None:
      self.random_producers = random_producers(self.n)
    elif len(_random_producers) != self.n:
      raise Exception("len(random_producer) != n")
    else:
      self.random_producers = _random_producers

    self.expected_max_reward_id = self.random_producers.expected_max_reward_id

  def run(self):
    return self.random_producers.get_random()

class n_armed_bandit_problem_solver():
    def __init__(self,n,Q_initial,epsilon=0.01,alpha_decay=0.995):
      self.n = n
      self.epsilon = epsilon
      self.Q_table = np.squeeze(Q_initial)
      self.alpha = 0.1
      self.alpha_decay = 1
    def choose_action(self):
      if np.random.uniform(0,1) < self.epsilon:
        return np.random.randint(0,self.n)
      max_index = np.argmax(self.Q_table)
      max_indices = np.ndarray.flatten(np.nonzero(abs(self.Q_table - self.Q_table[max_index]) < 0.0001 )[0])
      return max_indices[np.random.randint(0,np.shape(max_indices)[0])]

    def update(self,action,reward):
      self.Q_table[action] = self.Q_table[action] *  (1 - self.alpha) +  (reward) * self.alpha
      self.alpha *= self.alpha_decay

def choose_return_and_update(bandit,solver):
  res = bandit.run()
  max_reward = np.max(res)
  min_reward = np.min(res)

  action = solver.choose_action()
  reward = res[action]
  solver.update(action,reward)
  ran_reward = res[np.random.randint(0,len(res))]

  return reward,max_reward,ran_reward


if __name__ == '__main__':
  round = 100000
  n = 10
  bandit = n_armed_bandit(n)
  solver0 = n_armed_bandit_problem_solver(n,100*np.ones(n),0)
  solver1 = n_armed_bandit_problem_solver(n,np.zeros(n), 0.01)
  solver2 = n_armed_bandit_problem_solver(n,np.zeros(n), 0.1)

  proportion0 = []
  counter0 = np.array([0.,0.,0.])
  proportion1 = []
  counter1 = np.array([0.,0.,0])
  proportion2 = []
  counter2 = np.array([0.,0.,0])
  proportion3 = []
  for i in range(round):

    counter0+=choose_return_and_update(bandit, solver0)
    counter1+=choose_return_and_update(bandit,solver1)
    counter2+=choose_return_and_update(bandit, solver2)
    if i == 0: continue
    #print("i",i,counter0,counter1,counter2)
    proportion0.append(counter0[0]/(counter0[1]))
    proportion1.append(counter1[0]/(counter1[1]))
    proportion2.append(counter2[0]/(counter2[1]))

    proportion3.append(counter0[2]/(counter0[1]))
  plt.figure(1)
  plt.plot(np.array(proportion0), c='r', label='p0')
  plt.plot(np.array(proportion1), c='g', label='eps=0.01')
  plt.plot(np.array(proportion2), c='b', label='eps=0.1')
  plt.plot(np.array(proportion3), c='y', label='random choices')
  plt.legend(loc='best')
  plt.xlabel("Steps")
  plt.ylabel("%\n Optimal action")
  plt.show()