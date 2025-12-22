import argparse
import os
import sys
import numpy as np
import pandas as pd

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

class MyEpsilonGreedy:

    
    def __init__(self, initial_epsilon=0.05, min_epsilon=0.005, decay=1.0):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_values, action_space):
        if np.random.random() < self.epsilon:
            return action_space.sample() 
        else:
            return np.argmax(q_values)
    def update(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


class CustomQLAgent:
    
    def __init__(self, starting_state, state_space, action_space, alpha=0.1, gamma=0.99, 
                 exploration_strategy=None):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy
        
        self.q_table = {}
        self.last_action = None

        self._init_state_q_values(starting_state)
    
    def _init_state_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
    
    def act(self):
        self._init_state_q_values(self.state)
        q_values = self.q_table[self.state]
        
        action = self.exploration_strategy.choose(q_values, self.action_space)
        self.last_action = action
        return action
    
    def learn(self, next_state, reward):
        self._init_state_q_values(self.state)
        self._init_state_q_values(next_state)
        
        current_q = self.q_table[self.state][self.last_action]
        next_max_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[self.state][self.last_action] = new_q
        
        self.state = next_state
        self.exploration_strategy.update()


if __name__ == "__main__":

    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 4
    episodes = 4

    env = SumoEnvironment(
        net_file="/Users/trueuser/Desktop/RL/hw1/sumo-rl/sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="/Users/trueuser/Desktop/RL/hw1/sumo-rl/sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=False,
        num_seconds=3600,
        min_green=5,
        delta_time=5,
    )
    
    
    for run in range(1, runs + 1):
        print(f"Run {run}/{runs}")
        initial_states = env.reset()
        
        ql_agents = {
            ts: CustomQLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=MyEpsilonGreedy(
                    initial_epsilon=0.05,
                    min_epsilon=0.005,
                    decay=decay
                ),
            )
            for ts in env.ts_ids
        }
        
        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)
            
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                next_states, rewards, done, info = env.step(action=actions)
                
                for agent_id in next_states.keys():
                    ql_agents[agent_id].learn(
                        next_state=env.encode(next_states[agent_id], agent_id),
                        reward=rewards[agent_id]
                    )
            
            env.save_csv(f"outputs/4x4/custom_ql-4x4grid_run{run}", episode)
            print(f"  Episode {episode}/{episodes} completed")
    
    env.close()