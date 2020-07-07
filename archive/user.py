from helpers import *
from configurations import Config

c = Config()

class User(object):

    def __init__(self):
        self.max_round = c.constants['run']['max_rounds']

    def reset(self):
        return self.get_response()
    
    def get_response(self):
        response = {'intent':'','inform_slots':{},'request_slots':{}}
        while True:
            c1, c2 = True, True
            string = input('response: ')
            chunks = string.split('/')
            c0 = chunks[0] in c.usim_intents
            response['intent'] = chunks[0]
            for inform in filter(None,chunks[1].split(',')):
                key,val = inform.split(':')
                if key not in c.all_slots:
                    c1 = False
                    break
                response['inform_slots'][key] = val
            for request in filter(None,chunks[2].split(',')):
                if request not in c.all_slots: 
                    c2 = False
                    break
                response['request_slots'][request] = 'UNK'
            if c0 and c1 and c2: 
                break
        return response

    def get_success(self, success=None):
        while success not in (-1,0,1):
            success = int(input('success: '))
        return success

    def step(self, agent_action):
        print('agent action:', agent_action)
        response = {'intent':'','request_slots':{},'inform_slots':{}}
        if agent_action['round'] == self.max_round:
            response['intent'] = 'done'
            success = c.FAIL
        else:
            response = self.get_response()
            success = self.get_success()
        return response, get_reward(success,self.max_round), success in (c.FAIL,c.SUCCESS), success == 1