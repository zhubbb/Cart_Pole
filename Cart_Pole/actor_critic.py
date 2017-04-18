import os

import gym

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.contrib.layers import *
import numpy as np
import matplotlib

import collections
import itertools
import sys

if "../" not in sys.path:
    sys.path.append("../")

with tf.device('/cpu:0'):
    class Actor_policyEstimator():

        def __init__(self, env, alpha=0.001, scope="Actor"):
            NUM_INTPUT_FEATURES = env.observation_space.shape[0]

            try:
                output_units = env.action_space.shape[0]
            except AttributeError:
                output_units = env.action_space.n
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, shape=[NUM_INTPUT_FEATURES], name="state")  # [] means scalar
                self.action = tf.placeholder(tf.int32, [], "action")
                self.target_advantage = tf.placeholder(tf.float32, [], "target")

                weight_init = tf.contrib.layers.xavier_initializer(seed=13234, uniform=False)
                self.hidden = fully_connected(
                    inputs=tf.expand_dims(self.state, axis=0),
                    num_outputs=20,
                    activation_fn=tf.nn.tanh,
                    weights_initializer=weight_init,
                    weights_regularizer=None,
                    biases_initializer=0,
                )
                self.output = fully_connected(
                    inputs=self.hidden,
                    num_outputs=output_units,
                    activation_fn=None,
                    weights_initializer=weight_init,
                    weights_regularizer=None,
                    biases_initializer=0,
                )
                # action_probs
                self.pi = tf.squeeze(tf.nn.softmax(self.output))
                self.pi_action = tf.contrib.distributions.Bernoulli(tf.gather(self.pi, 1)).sample()
                self.pi_given_a = tf.gather(self.pi, self.action)

                # Loss
                self.eligibility = tf.log(self.pi_given_a) * self.target_advantage
                self.loss = -self.eligibility

                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
                self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        def predict(self, state, sess=None):
            sess = sess or tf.get_default_session()
            return sess.run(self.pi, {self.state: state})

        def getaction(self, state, sess=None):
            sess = sess or tf.get_default_session()
            return sess.run(self.pi_action, {self.state: state})

        def update(self, state, action, advantage, sess=None):
            sess = (sess or tf.get_default_session())
            feed_dict = {self.state: state, self.target_advantage: advantage, self.action: action}
            _, loss = sess.run([self.train_op, self.loss], feed_dict)
            return loss

with tf.device('/cpu:0'):
    class Critic_ValueEstimator:
        def __init__(self, env, alpha=0.001, scope="Critic"):
            NUM_INTPUT_FEATURES = env.observation_space.shape[0]

            output_units = 1
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, shape=[NUM_INTPUT_FEATURES], name="state")
                self.target = tf.placeholder(tf.float32, [], "target")
                weight_init = tf.contrib.layers.xavier_initializer(seed=1234, uniform=False)
                self.hidden = fully_connected(
                    inputs=tf.expand_dims(self.state, axis=0),
                    num_outputs=20,
                    activation_fn=tf.nn.tanh,
                    weights_initializer=weight_init,
                    weights_regularizer=None,
                    biases_initializer=weight_init,
                )
                self.output = fully_connected(
                    inputs=self.hidden,
                    num_outputs=output_units,
                    activation_fn=None,
                    weights_initializer=weight_init,
                    weights_regularizer=None,
                    biases_initializer=weight_init,
                )
            self.utility = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.utility, self.target)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        def predict(self, state, sess=None):
            sess = sess or tf.get_default_session()
            return sess.run(self.utility, {self.state: state})

        def update(self, state, target, sess=None):
            sess = sess or tf.get_default_session()
            feed_dict = {self.state: state, self.target: target}
            _, loss = sess.run([self.train_op, self.loss], feed_dict)
            return loss

with tf.device('/cpu:0'):
    def actor_critic(env, actor, critic, num_ep, discount_factor=0.998):
        
        aciton_list = np.arange(env.action_space.n)
        MEM = 5
        track_returns = []
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        train = True
        advantage_factor=10 # to enlarge the effect of advantage to facilitate training
        for ep in range(num_ep):
            state = env.reset()
            episodes = []
            I = 1.0
            # One step
            states = []
            targets = []
            actions = []
            stats_episode_rewards=0
            for t in itertools.count():
                # take a step
                # print(t)
                action_prob = actor.predict(state)
                action = np.argmax(action_prob)
                action = np.random.choice(aciton_list, p=action_prob)
                # action = actor.getaction(state)

                next_state, reward, done, info = env.step(action)
                episodes.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))
                # record stats
                stats_episode_rewards += reward
                
                #time = t / 100 - 1.0
                I *= discount_factor
                if done and t < 200:  # stop at 200 since we dont know
                    value_next = 0
                else:
                    value_next = critic.predict(next_state)

                target = reward + value_next * discount_factor
                target_advantage = (target - critic.predict(state))*advantage_factor
                # target_advantage=target_advantage*abs(target_advantage)/100
                # update
                if train:
                    critic.update(state, I * target)
                    # I is used to reduce learning rate when time increases and the result reaching equilibrium
                    if ep >= 200:
                        actor.update(state, action, I * target_advantage)

                states.append(state)
                targets.append(I * target_advantage)
                actions.append(action)

                if done:
                    break;
                state = next_state

            # for i in range(t+1):
            #     state=states[i]
            #     action=actions[i]
            #     target=targets[i]
            #     actor.update(state,action,target)


            track_returns.append(np.sum([ep.reward for ep in episodes]))
            track_returns = track_returns[-MEM:]
            mean_return = np.mean(track_returns)
            if mean_return > 198:
                print("reach {} ,stop training".format(mean_return))
                train = False  # to reduce randomness that lead to oscillation
            else:
                train=True
            print("Step {} @ Episode {}/{} ({})".format(
                t, ep + 1, num_ep, stats_episode_rewards), end="\n", flush=False)

        return stats_episode_rewards

with tf.device('/cpu:0'):
    env = gym.make('CartPole-v0')
    tf.reset_default_graph()

    global_step = tf.Variable(tf.constant(0, dtype=tf.int32, shape=[]), name="global_step", trainable=False)
    actor = Actor_policyEstimator(env)
    critic = Critic_ValueEstimator(env)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # global_variable_initializer
        stats = actor_critic(env, actor, critic, 600)
       




