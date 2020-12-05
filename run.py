import numpy as np
from q_agent import Agent as QAgent


agent = QAgent(env_name='FrozenLake-v0', a=0.3)
training_returns = agent.play_n_episodes(5000, is_training=True)
policy_returns = agent.play_n_episodes(1000, is_training=False)
print('Average return of final policy for FrozenLake:', np.mean(policy_returns))

_8x8_agent = QAgent(env_name='FrozenLake8x8-v0', a=0.1, e_step=0.000001)
_8x8_training_returns = _8x8_agent.play_n_episodes(25000, is_training=True)
_8x8_policy_returns = _8x8_agent.play_n_episodes(1000, is_training=False)
print('Average return of final policy for FrozenLake 8x8:', np.mean(_8x8_policy_returns))
