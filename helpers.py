def remove_empty_slots(d):
    for i in d.keys():
        d[i] = {k:v for k,v in d[i].items() if v}
        
def list_to_dict(l):
    return {key: index for index, key in enumerate(l)}

def get_reward(success, max_round):
	reward = -1
	if success == -1:
		reward += -max_round
	if success == 1:
		reward += 4*max_round
	return reward
