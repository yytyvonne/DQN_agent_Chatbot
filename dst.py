import copy
import numpy as np
from helpers import *
from collections import defaultdict
from configurations import Config 

c = Config()

class DatabaseQuery:
    def __init__(self, database):
        self.db = database
        self.db_index = self.index(self.db)
        self.no_query = c.no_query_keys
        self.match_key = c.usim_def_key

    def index(self, database):
        index = defaultdict(set)
        for i,vdict in database.items():
            for k,v in vdict.items():
                index[(k,v.lower())].add(i)
        return index
    
    def stat_vals(self, key, res):
        stat = defaultdict(int)
        for i,vdict in res.items():
            if key in vdict:
                stat[vdict[key]] += 1
        return stat

    def fill_slot(self, informs, slot):
        key = list(slot.keys())[0]
        res = self.find_recomm({k:v for k,v in informs.items() if k != key})
        fill,stat = {}, self.stat_vals(key,res)
        fill[key] = informs[key] = max(stat,key=stat.get) if stat else 'no match'
        return fill

    def find_recomm(self, informs):
        options = set(self.db.keys())
        for k,v in informs.items():
            if k in self.no_query or v=='anything': 
                continue
            options = list(self.db_index[(k,v.lower())].intersection(options))
        return {i:self.db[i] for i in options}

    def stat_recomm(self, informs):
        stat = defaultdict(int)
        for k,v in informs.items():
            if k in self.no_query: 
                stat[k] = 0
                continue
            stat[k] = len(self.db) if v == 'anything' else len(self.db_index[(k,v.lower())])
        stat['match_all'] = len(self.find_recomm(informs))
        return stat

class DST(object):
    
    def __init__(self, database):
        self.dbq = DatabaseQuery(database)
        self.match_key = c.usim_def_key
        self.intents_dict = list_to_dict(c.all_intents)
        self.slots_dict = list_to_dict(c.all_slots)
        self.max_rounds = c.constants['run']['max_rounds']
        self.state_size = 2*len(self.intents_dict)+7*len(self.slots_dict)+3+self.max_rounds
        self.none_state = np.zeros(self.state_size)
        self.reset()
    
    def reset(self):
        self.informs = {}
        self.history = []
        self.round_num = 0

    def print_history(self):
        for action in self.history: 
            print(action)

    def update_state_agent(self, agent_action):
        if agent_action['intent'] == 'inform':
            agent_action['inform_slots'] = self.dbq.fill_slot(self.informs,agent_action['inform_slots'])
        if agent_action['intent'] == 'match_found':
            results = self.dbq.find_recomm(self.informs)
            if results:
                key,value = list(results.items())[0] # random pick
                agent_action['inform_slots'] = copy.deepcopy(value)
                agent_action['inform_slots'][self.match_key] = str(key)
            else:
                agent_action['inform_slots'][self.match_key] = 'no match'
            self.informs[self.match_key] = agent_action['inform_slots'][self.match_key]
        agent_action.update({'round':self.round_num,'speaker':'Agent'})
        self.history.append(agent_action)

    def update_state_user(self, user_action):
        for key, value in user_action['inform_slots'].items():
            self.informs[key] = value
        user_action.update({'round': self.round_num,'speaker': 'User'})
        self.history.append(user_action)
        self.round_num += 1

    def get_state(self, done=False):
        if done: 
            return self.none_state
        user_action = self.history[-1]
        m_stat = self.dbq.stat_recomm(self.informs)
        agent_action = self.history[-2] if len(self.history) > 1 else None
        user_act_rep = np.zeros((len(self.intents_dict),))
        user_inf_slots_rep = np.zeros((len(self.slots_dict),))
        user_req_slots_rep = np.zeros((len(self.slots_dict),))
        user_act_rep[self.intents_dict[user_action['intent']]] = 1.0
        for key in user_action['inform_slots'].keys():
            user_inf_slots_rep[self.slots_dict[key]] = 1.0
        for key in user_action['request_slots'].keys():
            user_req_slots_rep[self.slots_dict[key]] = 1.0
        agent_action_rep = np.zeros((len(self.intents_dict),))
        agent_inf_slots_rep = np.zeros((len(self.slots_dict),))
        agent_req_slots_rep = np.zeros((len(self.slots_dict),))
        if agent_action:
            agent_action_rep[self.intents_dict[agent_action['intent']]] = 1.0
            for key in agent_action['inform_slots'].keys():
                agent_inf_slots_rep[self.slots_dict[key]] = 1.0
            for key in agent_action['request_slots'].keys():
                agent_req_slots_rep[self.slots_dict[key]] = 1.0
        current_slots_rep = np.zeros((len(self.slots_dict),))
        if agent_action:
            for key in self.informs:
                current_slots_rep[self.slots_dict[key]] = 1.0
        turn_rep = np.zeros((1,))+self.round_num/5. #1 hot
        turn_1h_rep = np.zeros((self.max_rounds,))
        turn_1h_rep[self.round_num-1] = 1.0
        kb_cnt_rep = np.zeros((len(self.slots_dict)+1,))+m_stat['match_all']/100.
        for key in m_stat.keys():
            if key in self.slots_dict: 
                kb_cnt_rep[self.slots_dict[key]] = m_stat[key]/100.
        kb_bin_rep = np.zeros((len(self.slots_dict)+1,))+np.sum(m_stat['match_all']>0.)
        for key in m_stat.keys():
            if key in self.slots_dict: kb_bin_rep[self.slots_dict[key]] = np.sum(m_stat[key]>0.)
        state_representation = np.hstack([user_act_rep,user_inf_slots_rep,user_req_slots_rep,
							   agent_action_rep,agent_inf_slots_rep,agent_req_slots_rep,
							   current_slots_rep,turn_rep,turn_1h_rep,kb_cnt_rep,kb_bin_rep]).flatten()
        return state_representation