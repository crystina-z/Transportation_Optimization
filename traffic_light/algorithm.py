'''
This file contains
 -  Neural Network Structure
 -  Training Process
 -  helper composition ( buffer, target network update, etc)
'''

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
import matplotlib.pyplot as plt
import os
from environment import gameEnv

import sys
import optparse
import subprocess
import random

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

env = gameEnv()

''' feature of input matrix '''
ELEMENT_NUM = 120
IMAGE_X = 12
IMAGE_Y = 10
IMAGE_Z = 1

'''
    Qnetwork defines the neural network structure
    the network is a transfromed CNN, with two convolution layers and two parallel fully connected layers streamA and streamV
    where streamA represent Advantage streamA and streamV represent Value function stream

    Input of network: current state, in the form of a 2D matrix
    Output:
        Qout: a vector of Q-value, one for each action
        predict: a scalar, range from 0 to 3, each representing an action (a phase of traffic light)
'''
class Qnetwork():
    def __init__(self, h_size):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through two convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, ELEMENT_NUM], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, IMAGE_X, IMAGE_Y, IMAGE_Z])
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn, num_outputs=32, kernel_size=[3, 3], stride=[2, 2], padding='VALID',
            biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1, num_outputs=h_size, kernel_size=[5, 4], stride=[1, 1], padding='VALID',
            biases_initializer=None)


        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv2, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

'''
    Helper class, used to manage trained experiments and provide interface for Q-network
    form of each experience: (s, r, a, s', d)
    s: current state
    r: rewards
    a: action taken
    s': next state
    d: if current episode is done or not

    Provided Function:
    add(self, experience):
        add new experience into buffer
        if buffer run out of space, then reserve the latest buffer_size experience

    sample(self, size):
        generate 'size' number of training data
'''
class experience_buffer():
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


# reshape the states into [scalarInput] shape;
def processState(states):
    return np.reshape(states, [ELEMENT_NUM])

# Update the parameters of our target network with those of the primary network
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

# helper function, used to update target network
def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


batch_size = 32  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .95 # Discount factor on the target Q-values
startE = 1 # Starting chance of random action
endE = 0.1 # Final chance of random action
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 100 # How many episodes of game environment to train network with.
pre_train_steps = 3000 # How many steps of random actions before training begins.
max_epLength = 3000 # The max allowed length of our episode.
load_model = False # Whether to load a saved model.
path = "./dqn" # The path to save our model to.
h_size = 32 # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 # Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / annealing_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

'''
our training process
'''
# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    f = open('record.txt','w')

    for i in range(num_episodes):
        print (" --- in the ", i, "th episodes ---")
        if i % 50 == 0:
            env.generate_routefile()

        episodeBuffer = experience_buffer()
        # Reset environment and get first new observation
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        # The Q-Network
        while j < max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                #print ("random choose action")
                a = np.random.randint(0, 4)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            s1, r, d = env.stepForward(a)
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(
                np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                                 feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})
                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.
            rAll += r
            s = s1


            if d == True:
                break

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)

        # Periodically save the model.
        result = env.sum/env.step
        print(result)
        f.write('%f\n'% result)

        if i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print("----------", total_steps, np.mean(rList[-10:]), e, "----------")
    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
    env.close()
    f.close()


rMat = np.resize(np.array(rList), [len(rList)//100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)
plt.savefig("imagineHack.png")
plt.show()


''' tesing process '''
'''
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./dqn/model-99.ckpt.meta')
    saver.restore(sess, path + '/model-99.ckpt')

    s = env.reset()
    s = processState(s)
    sess.run(tf.global_variables_initializer())
    a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
    print(a)

    num_episodes = 5
    max_epLength = 3000
    total_steps = 0

    for i in range(num_episodes):
        print(i, "th episode")
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        loss = 0

        while j < max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j += 1
            # a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            s1, r, d = env.stepForward(a)
            s = s1
            s = processState(s)
            total_steps += 1
            rAll += r
            if d == True:
                break

        rList.append(rAll)

rMat = np.resize(np.array(rList), [len(rList), 1])
rMean = np.average(rMat, 1)
plt.plot(rMean)
plt.savefig('testing_result.png')
'''
