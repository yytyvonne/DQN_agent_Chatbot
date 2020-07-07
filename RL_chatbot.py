import pickle
from configurations import Config
from helpers import *
from dqn_agent import DQN as DQN
from dqn_agent import DDQN as DDQN
from dqn_agent import PerDQN as PerDQN
from dst import *
from emc import *
from user import *
from user_simulator import *

c = Config()

class ChatBot(object):
    def __init__(self, model = 0):
        self.movie_db = pickle.load(open('data/movie_db.pkl', 'rb'), encoding='latin1')
        self.movie_dict = pickle.load(open('data/movie_dict.pkl', 'rb'), encoding='latin1')
        self.user_goals = pickle.load(open('data/movie_user_goals.pkl', 'rb'), encoding='latin1')
        remove_empty_slots(self.movie_db)
        self.dst = DST(self.movie_db)
        self.emc = EMC(self.movie_dict)
        if model == 0:
            self.dqn = DQN(self.dst.state_size)
        elif model == 1:
            self.dqn = DDQN(self.dst.state_size)
        else:
            self.dqn = PerDQN(self.dst.state_size)
        
    def episode_reset(self): #conversation = episode
        self.dst.reset()
        self.dqn.reset()
        user_action = self.user.reset()
        self.emc.add_error(user_action)
        self.dst.update_state_user(user_action)
		
    def run(self, state, step,warmup=False):
        agent_index, agent_action = self.dqn.get_action(state,step,use_rule=warmup)
        self.dst.update_state_agent(agent_action)
        user_action, reward, done, success = self.user.step(agent_action)
        if not done: 
            self.emc.add_error(user_action)
        self.dst.update_state_user(user_action)
        next_state = self.dst.get_state(done)
        self.dqn.add_experience(state, agent_index, reward, next_state, done)
        return next_state, reward, done, success

    def warmup(self, total_steps=0):
        print("warm up: start")
        while total_steps < c.constants['run']['warmup_mem'] and not self.dqn.is_memory_full():
            self.episode_reset()
            state = self.dst.get_state()
            done = False
            while not done:
                next_state,_,done,_ = self.run(state,total_steps,warmup=True)
                state = next_state
                total_steps += 1
        print("warm up: end")

    def train(self):
        print('training start')
        self.user = UserSimulator(self.movie_db, self.user_goals)
        self.warmup()
        episode = 0
        success_total = 0
        reward_total = 0
        best_success_rate = 0.0
        success_rates = {}
        rwds = {}
        while episode < c.constants['run']['ep_run_num']:
            self.episode_reset()
            episode += 1
            done = False
            state = self.dst.get_state()
            total_steps=0
            while not done:
                next_state, reward, done, success = self.run(state, total_steps)
                total_steps += 1
                reward_total += reward
                state = next_state
            success_total += success
            if episode%c.constants['run']['train_freq'] == 0:
                success_rate = success_total/c.constants['run']['train_freq']
                average_reward = reward_total/c.constants['run']['train_freq']
                print('Episode:',episode, 'best success rate:', success_rate, 'average reward:',average_reward)
                success_rates[episode] = success_rate
                rwds[episode] = average_reward
                if success_rate >= best_success_rate and success_rate >= c.constants['run']['succ_thres']:
                    self.dqn.reset_memory()
                    best_success_rate = success_rate
                    self.dqn.save_weights()
                success_total, reward_total = 0,0
                self.dqn.copy()
                self.dqn.train()
        print('training end')
        return success_rates, rwds

    def test(self):
        print('testing start')
        self.user = UserSimulator(self.movie_db, self.user_goals)
        episode = 0
        success_total = 0
        reward_total = 0
        while episode < c.constants['run']['ep_run_num']:
            self.episode_reset()
            episode += 1
            ep_reward = 0
            done = False
            state = self.dst.get_state()
            total_steps=0
            while not done:
                agent_index, agent_action = self.dqn.get_action(state, total_steps)
                total_steps += 1
                self.dst.update_state_agent(agent_action)
                user_action, reward, done, success = self.user.step(agent_action)
                ep_reward += reward
                if not done: 
                    self.emc.add_error(user_action)
                self.dst.update_state_user(user_action)
            print('Episode: {} Success: {} Reward: {}'.format(episode, success, ep_reward))
            success_total += success
            reward_total += ep_reward
        print('testing end')
        return success_total, reward_total
        
        
        
        
        