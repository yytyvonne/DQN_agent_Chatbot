import tensorflow as tf
import random
import numpy as np
import re
import copy
from configurations import Config 
import math
np.random.seed(1234)

c = Config()

class RL(object):
    def __init__(self, state_size):
        for key, value in c.constants['agent'].items():
            setattr(self, key, value)
        
        self.memory = []
        self.memory_index = 0
        self.state_size = state_size
        self.actions = c.agent_actions
        self.action_n = len(self.actions)
        self.rule_requests = c.rule_requests
        self.build_model()
        self.load_weights()
        self.reset()
        
    def build_model(self):
        pass
    
    def save_weights(self):
        pass
    
    def load_weights(self):
        pass
    
    def reset(self):
        self.rule_current_slot_index = 0
        self.rule_done = False
        
    def index_action(self, index):
        for i, action in enumerate(self.actions):
            if index == i:
                return copy.deepcopy(action)
    
    def action_index(self, a):
        for i, action in enumerate(self.actions):
            if action == a:
                return i
        
    def rule_action(self):
        response = {'intent': '', 'inform_slots': {}, 'request_slots': {}}
        if self.rule_current_slot_index < len(self.rule_requests):
            response['intent'] = 'request'
            response['request_slots'] = {self.rule_requests[self.rule_current_slot_index]: 'UNK'}
            self.rule_current_slot_index += 1
        elif not self.rule_done:
            response['intent'] = 'match_found'
            self.rule_done = True
        elif self.rule_done:
            response['intent'] = 'done'
        return self.action_index(response), response
            
    def dqn_action(self, state):
        pass

    def get_action(self, state,step,use_rule=False): #use rule=True --> random policy  
        #if not use_rule:
            #self.epsilon = self.min_epsilon + math.exp(-self.lamda * step) * (self.max_epsilon - self.min_epsilon) 
        if self.epsilon > random.random():
            index = random.randint(0,self.action_n-1)
            return index, self.index_action(index)
        return self.rule_action() if use_rule else self.dqn_action(state)

    def add_experience(self, state, action, reward, nstate, done):
        pass
    
    def reset_memory(self):
        self.memory = []
        self.memory_index = 0

    def is_memory_full(self):
	    return len(self.memory) == self.max_memory_size

    def copy(self):
        pass
		
    def train(self):
        pass

load_weights = "load_path"
save_weights = "save_path"

class DQN(RL):

    def __init__(self, state_size):        
        super(DQN, self).__init__(state_size)
        target_params = tf.get_collection('target_params')
        behaviour_params = tf.get_collection('behaviour_params')
        self.optimized = [tf.assign(target,behaviour) for target, behaviour in zip(target_params, behaviour_params)]
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name = 'state')
        self.q_target = tf.compat.v1.placeholder(tf.float32,[None,self.action_n],name='q_target')
        self.next_state = tf.compat.v1.placeholder(tf.float32,[None,self.state_size],name='next_state')
        w_init = tf.random_normal_initializer(0,0.1)
        b_init = tf.constant_initializer(0.1)
        
        def build_layers(inputs, c_names):
            with tf.variable_scope('l1', reuse=tf.AUTO_REUSE):  #1st dense layer
                w1 = tf.get_variable('w1',[self.state_size,self.hidden_size],initializer=w_init,collections=c_names)
                b1 = tf.get_variable('b1',[1,self.hidden_size],initializer=b_init,collections=c_names)
                l1 = tf.nn.relu(tf.matmul(inputs,w1)+b1)  #activation relu
            with tf.variable_scope('l2', reuse=tf.AUTO_REUSE): #2nd dense layer
                w2 = tf.get_variable('w2',[self.hidden_size,self.action_n],initializer=w_init,collections=c_names)
                b2 = tf.get_variable('b2',[1,self.action_n],initializer=b_init,collections=c_names)
                out = tf.matmul(l1,w2)+b2 #linear activation
            return out
        with tf.variable_scope('behaviour_net', reuse=tf.AUTO_REUSE):		
            self.q_behaviour = build_layers(self.state,['behaviour_params',tf.GraphKeys.GLOBAL_VARIABLES])
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):	 
            self.loss = tf.losses.mean_squared_error(self.q_target,self.q_behaviour)
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):	
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        with tf.variable_scope('target_net', reuse=tf.AUTO_REUSE):	
            self.q_next = build_layers(self.next_state,['target_params',tf.GraphKeys.GLOBAL_VARIABLES])

    def dqn_action(self, state):
        state= np.array(state).reshape(1, len(state))
        index = np.argmax(self.sess.run(self.q_behaviour, feed_dict={self.state: state}))
        action = self.index_action(index)
        return index, action

    def add_experience(self, state, action, reward, next_state, done):
        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = (state, action, reward, next_state, done)
        self.memory_index = (self.memory_index + 1) % self.max_memory_size

    def copy(self):
        self.sess.run(self.optimized)
                
            
    def train(self):
        n_batch = math.floor(len(self.memory)/self.batch_size)
        for b in range(n_batch):
            batch = random.sample(self.memory,self.batch_size)
            states,actions, rewards, next_states, dones = zip(*batch)
            q_behaviour, q_next = self.sess.run([self.q_behaviour,self.q_next],feed_dict={self.state:np.array(states),self.next_state:np.array(next_states)})
            q_target = q_behaviour.copy()
            batch_index = np.arange(self.batch_size,dtype=np.int32)
            q_target[batch_index,np.array(actions)] = np.array(rewards)+self.gamma*np.max(q_next,axis=1)*(1-np.array(dones))
            _,cost = self.sess.run([self.train_op,self.loss],feed_dict={self.state:np.array(states),self.q_target:q_target})            
        
class DDQN(DQN):

	def __init__(self, state_size):
		super(DDQN,self).__init__(state_size)

	def train(self):
		n_batch = math.floor(len(self.memory)/self.batch_size) #minibatch
		for b in range(n_batch):
			batch = random.sample(self.memory,self.batch_size)
			states,actions,rewards,next_states,dones = zip(*batch)
			q_behaviour,q_next = self.sess.run([self.q_behaviour,self.q_next],feed_dict={self.state:np.array(states),self.next_state:np.array(next_states)})
			q_eval = self.sess.run(self.q_behaviour,feed_dict={self.state:np.array(next_states)})
			q_target = q_behaviour.copy()
			batch_index = np.arange(self.batch_size,dtype=np.int32)
			q_target[batch_index,np.array(actions)] = np.array(rewards)+self.gamma*q_next[batch_index,np.argmax(q_eval,axis=1)]*(1-np.array(dones))
			_,cost = self.sess.run([self.train_op,self.loss],feed_dict={self.state:np.array(states),self.q_target:q_target})        


class SumTree(object):
    write = 0

    def __init__(self, size):
        self.size = size
        self.tree = np.zeros( 2*size - 1 )
        self.data = []

    def propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s-self.tree[left])

    def total(self):
        points = self.tree[0]
        return points

    def add(self, p, data):
        if len(self.data) < self.size:
            self.data.append(None)
        idx = self.write + self.size - 1
        self.data[self.write] = data
        self.update(p, idx)

        self.write += 1
        if self.write >= self.size:
            self.write = 0

    def update(self, p, idx):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self.propagate(idx, change)

    def get(self, s):
        idx = self.retrieve(0, s)
        dataIdx = idx - self.size + 1

        return (idx, self.tree[idx], self.data[dataIdx])
    
class Memory(object):
    e = 0.01
    alpha = 0.6
    beta = 0.4
    beta_incre_per_sampling = 0.001
    abs_error_upper = 1.
    
    def __init__(self, size):
        self.tree = SumTree(size)
        
    def store(self, trans):
        max_p = np.max(self.tree.tree[-self.tree.size:])
        if max_p == 0:
            max_p = self.abs_error_upper
        self.tree.add(max_p,trans)
        
    def sample(self, n):
        self.beta = np.min([1,self.beta+self.beta_incre_per_sampling])
        ids = []
        mem = []
        is_wg = np.empty((n,1))
        pt = self.tree.tree[0]
        p_seg = pt / n 
        min_prob = np.min(self.tree.tree[-self.tree.size:-self.tree.size+len(self.tree.data)])/pt
        for i in range(n):
            trid,p,data = self.tree.get(np.random.uniform(p_seg*i,p_seg*(i+1)))
            is_wg[i,0] = np.power(p/self.tree.tree[0]/min_prob,-self.beta)
            ids.append(trid)
            mem.append(data)
        return np.array(ids), mem, is_wg
    
    def batch_update(self, trids, abs_error):
        ps = np.power(np.minimum(abs_error+self.e,self.abs_error_upper),self.alpha)
        for trid,p in zip(trids,ps):
            self.tree.update(p,trid)

class PerDQN(DQN):

    def __init__(self, state_size):
        super(PerDQN,self).__init__(state_size)
        self.memory = Memory(self.max_memory_size)
        
    def build_model(self):
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name = 'state')
        self.q_target = tf.compat.v1.placeholder(tf.float32,[None,self.action_n],name='q_target')
        self.is_wg = tf.placeholder(tf.float32,[None,1],name='is_wg')
        self.next_state = tf.compat.v1.placeholder(tf.float32,[None,self.state_size],name='next_state')
        w_init = tf.random_normal_initializer(0,0.1)
        b_init = tf.constant_initializer(0.1)
        
        def build_layers(inputs, c_names, trainable):
            with tf.variable_scope('l1', reuse=tf.AUTO_REUSE):  #1st dense layer
                w1 = tf.get_variable('w1',[self.state_size,self.hidden_size],initializer=w_init,collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1',[1,self.hidden_size],initializer=b_init,collections=c_names, trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(inputs,w1)+b1)  #activation relu
            with tf.variable_scope('l2', reuse=tf.AUTO_REUSE): #2nd dense layer
                w2 = tf.get_variable('w2',[self.hidden_size,self.action_n],initializer=w_init,collections=c_names,trainable=trainable)
                b2 = tf.get_variable('b2',[1,self.action_n],initializer=b_init,collections=c_names,trainable=trainable)
                out = tf.matmul(l1,w2)+b2 #linear activation
            return out
        with tf.variable_scope('behaviour_net', reuse=tf.AUTO_REUSE):		
            self.q_behaviour = build_layers(self.state,['behaviour_params',tf.GraphKeys.GLOBAL_VARIABLES],True)
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):	
            self.abs_error = tf.reduce_sum(tf.abs(self.q_target-self.q_behaviour),axis=1)
            self.loss = tf.reduce_mean(self.is_wg * tf.squared_difference(self.q_target,self.q_behaviour))
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):	
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        with tf.variable_scope('target_net', reuse=tf.AUTO_REUSE):	
            self.q_next = build_layers(self.next_state,['target_params',tf.GraphKeys.GLOBAL_VARIABLES],False)

    def add_experience(self, state, action, reward, nstate, done):
        self.memory.store((state,action,reward,nstate,done))

    def reset_memory(self):
        self.memory = Memory(self.max_memory_size)

    def is_memory_full(self):
        return len(self.memory.tree.data) == self.max_memory_size

    def train(self):
        n_batch = math.floor(len(self.memory.tree.data)/self.batch_size) #minibatch
        for b in range(n_batch):
            trids, batch, is_wg = self.memory.sample(self.batch_size)
            states,actions,rewards,next_states,dones = zip(*batch)
            q_behaviour,q_next = self.sess.run([self.q_behaviour,self.q_next],feed_dict={self.state:np.array(states),self.next_state:np.array(next_states)})
            q_eval = self.sess.run(self.q_behaviour,feed_dict={self.state:np.array(next_states)})
            q_target = q_behaviour.copy()
            batch_index = np.arange(self.batch_size,dtype=np.int32)
            q_target[batch_index,np.array(actions)] = np.array(rewards) + self.gamma * q_next[batch_index,np.argmax(q_eval,axis=1)]* (1-np.array(dones))
            _, abs_error, cost = self.sess.run([self.train_op,self.abs_error,self.loss],feed_dict={self.state:np.array(states),self.q_target:q_target,self.is_wg:is_wg})
            self.memory.batch_update(trids,abs_error)
            

    
    
    
    