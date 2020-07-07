import random
import copy
from helpers import *
from configurations import Config

c = Config()

class UserSimulator(object):
    #The main difference between this system and a real user is that the user might change their mind as they learn more about what tickets are available.

    def __init__(self, database, user_goals):
        self.db = database
        self.goals = user_goals
        self.def_key = c.usim_def_key
        self.informs = [c.usim_req_key]
        self.no_query = c.no_query_keys
        self.max_round = c.constants['run']['max_rounds']

    def reset(self):
        self.goal = random.choice(self.goals)
        self.goal['request_slots'][self.def_key] = 'UNK'
        self.state = {'intent':'','inform_slots':{},'request_slots' :{},'history_slots' :{},'rest_slots':{}}
        self.state['rest_slots'].update(self.goal['inform_slots'])
        self.state['rest_slots'].update(self.goal['request_slots'])
        self.check = c.FAIL
        return self.init_action()

    def init_action(self):
        def add_slot(key):
            self.state['rest_slots'].pop(key)
            self.state['inform_slots'][key] = self.state['history_slots'][key] = self.goal['inform_slots'][key]
        
        self.state['intent'] = 'request'
        if self.goal['inform_slots']:
            for key in self.informs:
                if key in self.goal['inform_slots']:
                    add_slot(key)
            if not self.state['inform_slots']:
                add_slot(random.choice(self.goal['inform_slots'].keys()))
        self.goal['request_slots'].pop(self.def_key)
        req_key = random.choice(list(self.goal['request_slots'].keys())) if self.goal['request_slots'] else self.def_key
        self.goal['request_slots'][self.def_key] = self.state['request_slots'][req_key] = 'UNK'
        response= {
			'intent'        :self.state['intent'],
			'inform_slots'  :copy.deepcopy(self.state['inform_slots']),
			'request_slots' :copy.deepcopy(self.state['request_slots'])}
        return response

    def step(self, agent_action):
        self.state['intent'] = ''
        self.state['inform_slots'].clear()
        done, success = False, c.NO_OUTCOME
        if agent_action['round'] == self.max_round:
            done, success = True, c.FAIL
            self.state['intent'] = 'done'
            self.state['request_slots'].clear()
        else:
            if agent_action['intent'] == 'inform':
                self.response_to_inform(agent_action)
            if agent_action['intent'] == 'request':
                self.response_to_request(agent_action)
            if agent_action['intent'] == 'match_found':
                self.response_to_match_found(agent_action)
            if agent_action['intent'] == 'done':
                done, success = True, self.response_to_done()
                self.state['intent'] = 'done'
                self.state['request_slots'].clear()
        response= {'intent':self.state['intent'],'inform_slots':copy.deepcopy(self.state['inform_slots']),'request_slots' :copy.deepcopy(self.state['request_slots'])}
        return response, get_reward(success,self.max_round), done, success == 1

    def response_to_inform(self, agent_action):
        key,val = list(agent_action['inform_slots'].items())[0]
        self.state['history_slots'][key] = val
        for name in ('rest_slots','request_slots'):
            if key in self.state[name]: 
                self.state[name].pop(key,None)
        if val != self.goal['inform_slots'].get(key,val):
            self.state['intent'] = 'inform'
            self.state['request_slots'].clear()
            self.state['inform_slots'][key] = self.state['history_slots'][key] = self.goal['inform_slots'][key]
        else:
            if self.state['request_slots']:
                self.state['intent'] = 'request'
            elif self.state['rest_slots']:
                def_in = self.state['rest_slots'].pop(self.def_key,False)
                if self.state['rest_slots']:
                    k,v = random.choice(list(self.state['rest_slots'].items()))
                    if v != 'UNK':
                        self.state['intent'] = 'inform'
                        self.state['rest_slots'].pop(k)
                        self.state['inform_slots'][k] = self.state['history_slots'][k] = v
                    else:
                        self.state['intent'] = 'request'
                        self.state['request_slots'][k] = 'UNK'
                else:
                    self.state['intent'] = 'request'
                    self.state['request_slots'][self.def_key] = 'UNK'
                if def_in == 'UNK':
                    self.state['rest_slots'][self.def_key] = 'UNK'
            else:
                self.state['intent'] = 'thanks'

    def response_to_request(self, agent_action):
        key = list(agent_action['request_slots'].keys())[0]
        if key in self.goal['inform_slots']:
            self.state['intent'] = 'inform'
            self.state['request_slots'].clear()
            self.state['rest_slots'].pop(key,None)
            self.state['inform_slots'][key] = self.state['history_slots'][key] = self.goal['inform_slots'][key]
        elif key in self.goal['request_slots'] and key in self.state['history_slots']:
            self.state['intent'] = 'inform'
            self.state['request_slots'].clear()
            self.state['inform_slots'][key] = self.state['history_slots'][key]
        elif key in self.goal['request_slots'] and key in self.state['rest_slots']:
            self.state['intent'] = 'request'
            self.state['request_slots'] = {key:'UNK'}
            rest_informs = {k:v for k,v in self.state['rest_slots'].items() if v != 'UNK'}
            if rest_informs:
                kc,vc = random.choice(list(rest_informs.items()))
                self.state['rest_slots'].pop(kc)
                self.state['inform_slots'][kc] = self.state['history_slots'][kc] = vc
        else:
            self.state['intent'] = 'inform'
            self.state['request_slots'].clear()
            self.state['inform_slots'][key] = self.state['history_slots'][key] = 'anything'

    def response_to_match_found(self, agent_action):
        agent_informs = agent_action['inform_slots']
        self.state['rest_slots'].pop(self.def_key,None)
        self.state['request_slots'].pop(self.def_key,None)
        self.state['history_slots'][self.def_key] = str(agent_informs[self.def_key])
        self.check = c.SUCCESS
        if agent_informs[self.def_key] == 'no match': 
            self.check = c.FAIL
        for key,val in self.goal['inform_slots'].items():
            if key in self.no_query: 
                continue
            if val != agent_informs.get(key,None):
                self.check = c.FAIL
                break
        if self.check == c.FAIL:
            self.state['intent'] = 'reject'
            self.state['request_slots'].clear()
        else:
            self.state['intent'] = 'thanks'

    def response_to_done(self):
        return c.FAIL if self.check == c.FAIL or self.state['rest_slots'] else c.SUCCESS
    
    
    
    