import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib.layers import *
np.random.seed(1)
tf.set_random_seed(1)
env = gym.make('CartPole-v0')
NUM_INPUT_FEATURES = env.observation_space.shape[0]
alpha = 0.01
gamma=0.999 # gamma has to be very very close to 1 like 0.999 to make it works, since it need to be long sighted to survive longer
x = tf.placeholder(tf.float32, (None, NUM_INPUT_FEATURES), 'x')
#a = tf.placeholder(tf.float32, name='a')

weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.01)

hidden_size = 300
output_size = 1

hidden1 = fully_connected(
    inputs=x,
    num_outputs=hidden_size,
    activation_fn=tf.nn.relu,
    weights_initializer=weight_init,
    weights_regularizer=None,
    biases_initializer=weight_init,
    scope='hidden1')

sigmoid = fully_connected(
    inputs=hidden1,
    num_outputs=output_size,
    activation_fn=tf.nn.sigmoid,
    weights_initializer=weight_init,
    weights_regularizer=None,
    biases_initializer=weight_init,
    scope='output')
# H = 20
# W1 = tf.get_variable("W1", shape=[4, H],
#            initializer=tf.contrib.layers.xavier_initializer())
# W2 = tf.get_variable("W2", shape=[H, 1],
#            initializer=tf.contrib.layers.xavier_initializer())
# layer1 = tf.nn.relu(tf.matmul(x, W1))
# score = tf.matmul(layer1,W2)
#sigmoid = tf.nn.sigmoid(score)

y=tf.placeholder(tf.float32, shape=(None,1),name = "input_y")
G_state=tf.placeholder(tf.float32)

pi= tf.contrib.distributions.Bernoulli(p=sigmoid)
pi_action = pi.sample()

pi_action_logprob=pi.log_prob(y)
#pi_action_logprob=y*tf.log(sigmoid) + (1-y)*tf.log(1-sigmoid)
cost = tf.reduce_mean(-1.0*pi_action_logprob*G_state) #must use reduce_mean to get it work


#optimizer = tf.train.AdamOptimizer(alpha)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEM=25
track_returns=[]
running_reward = None
for ep in range(100000000):
    obs=env.reset()
    G=0
    ep_actions=[]
    ep_rewards=[]
    xs = []
    Done=False
    t=0
    I=1
    while not Done:
        x_ = np.reshape(obs, [1,4])
        xs.append(x_)
        #env.render()
        #print(action)
        sig = sess.run(sigmoid, {x:x_})
        #action = 1 if np.random.uniform() < sig else 0
        action=sess.run(pi_action,{x:x_})[0][0]

        ep_actions.append(action)
        obs,reward,Done,info = env.step(action)
        reward*=I
        I*=gamma
        ep_rewards.append(reward)
        t+=1
    G_s0 = np.sum(ep_rewards)
    G_states = np.vstack(G_s0 - np.cumsum([0]+ep_rewards[:-1]))
    #G_states = discount_rewards(np.vstack(ep_rewards))
    #print("G ", G_states[2], G_states2[2])
    #assert(G_states[2] == G_states2[2])
    #print (np.array(ep_actions))
    print("-----------------")
    #print (np.vstack(ep_actions))
    actions = np.vstack(ep_actions)
    #actions = np.array(ep_actions)
    _, logProb = sess.run([train_op, pi_action_logprob] ,{x:np.vstack(xs),
        y:actions,
        G_state:G_states})
    print("log prob", logProb)

    track_returns.append(G_s0)
    track_returns = track_returns[-MEM:]
    mean_return = np.mean(track_returns)
    running_reward = G_s0 if running_reward is None else running_reward * 0.99 + G_s0 * 0.01
    print("Episode {} finished after {} steps with return {}".format(ep, t, G_s0))
    print("Mean return over the last {} episodes is {}".format(MEM,mean_return))
    print("cost {}".format(sess.run(cost,{x:np.array(np.vstack(xs)),
    y:actions,
    G_state:G_states})))

sess.close()
