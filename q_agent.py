import numpy as np
import gym
from tqdm import tqdm


class Agent:

    def __init__(self, env_name, a=0.3, e_start=1, e_min=0.1, e_step=0.00001, y=0.99):
        self.a = a
        self.e = e_start
        self.e_min = e_min
        self.e_step = e_step
        self.y = y

        self.env = gym.make(env_name)
        self.q = np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def random_action(self):
        return self.env.action_space.sample()

    def e_greedy_action(self, state):
        if np.random.rand() < self.e:
            return self.random_action()
        else:
            return self.best_action(state)

    def best_action(self, state):
        return np.argmax(self.q[state, :])

    def q_max(self, state):
        return np.max(self.q[state, :])

    def play_one_episode(self, is_training=True):
        state = self.env.reset()
        episode_return = 0
        done = False

        while not done:
            action = self.e_greedy_action(state) if is_training else self.best_action(state)
            self.e = max(self.e - self.e_step, self.e_min)

            next_state, reward, done, _ = self.env.step(action)
            episode_return += reward

            self.q[state, action] = self.q[state, action] + self.a * (reward + self.y * self.q_max(next_state) - self.q[state, action])
            state = next_state

        return episode_return

    def play_n_episodes(self, n, is_training=True):
        return [self.play_one_episode(is_training) for _ in tqdm(range(n))]
