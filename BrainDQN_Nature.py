# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.9  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100

try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply


class BrainDQN:

    def __init__(self, actions):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2,self.a = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T,self.aT= self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([5, 5, 7, 8])
        b_conv1 = self.bias_variable([8])

        W_conv2 = self.weight_variable([3, 3, 8, 16])
        b_conv2 = self.bias_variable([16])

        W_conv3 = self.weight_variable([3, 3, 16, 32])
        b_conv3 = self.bias_variable([32])

        W_fc1 = self.weight_variable([12 * 12 * 32 + 8, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # input layer

        stateInput = tf.placeholder("float", [None, 12, 12, 7])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 1) + b_conv1)
        #h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 1) + b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 12 * 12 * 32])

        actionnew = tf.placeholder("float", [None, 8])

        h_conv3_flat = tf.concat([h_conv3_flat, actionnew], 1)

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # Q Value layer
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

        return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2,actionnew

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-5).minimize(self.cost)

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        actionarray = []
        for i in range(0, BATCH_SIZE):
            actiontest = self.getwallstate(state_batch[i])
            actionarray.append(actiontest)

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch,
                                                    self.aT: actionarray})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch,
            self.a: actionarray
        })

        # save network every 100000 iteration
        if self.timeStep % 100 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn')

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, nextObservation, action, reward, terminal):
        newState = nextObservation.copy()
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        print("currentstat 0:", self.currentState[:, :, 0])
        print("currentstat 1:", self.currentState[:, :, 1])
        print("currentstat 2:", self.currentState[:, :, 2])
        print("currentstat 3:", self.currentState[:, :, 3])
        print("currentstat 4:", self.currentState[:, :, 4])
        print("currentstat 5:", self.currentState[:, :, 5])
        print("currentstat 6:", self.currentState[:, :, 6])

        print("action reward terminal",action,reward,terminal)

        print("newState 0:", newState[:, :, 0])
        print("newState 1:", newState[:, :, 1])
        print("newState 2:", newState[:, :, 2])
        print("newState 3:", newState[:, :, 3])
        print("newState 4:", newState[:, :, 4])
        print("newState 5:", newState[:, :, 5])
        print("newState 6:", newState[:, :, 6])


        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ STATE", state, \
              "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):

        print("getaction 0 ",self.currentState[:, :, 0])
        print("getaction 1 ", self.currentState[:, :, 1])
        print("getaction 2 ", self.currentState[:, :, 2])
        print("getaction 3 ", self.currentState[:, :, 3])
        print("getaction 4 ", self.currentState[:, :, 4])
        print("getaction 5 ", self.currentState[:, :, 5])
        print("getaction 6 ", self.currentState[:, :, 6])


        actionnew = self.getwallstate(self.currentState)
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState],
                                 self.a: [actionnew]})[0]
        print("Qvalue:",QValue)
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                print("Choose_action:", QValue)
                if action_index < 8:
                    #剔除
                    print("actionnew,action_index",actionnew,action_index)
                    while action_index < 8 \
                            and actionnew[action_index] == 0:
                        QValue[action_index] = -65535
                        action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing
            print("nothing")

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def setInitState(self, observation):
        self.currentState = observation

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 8个方向能否走 1能走，0不能走
    def getwallstate(self, stateInput):
        self.hight = 12
        actions = np.zeros(8)

        player_index = (np.argwhere(stateInput[:, :, 1] == 1)).tolist()[0]
        asix, asiy = player_index[0], player_index[1]
        #up
        if asix - 1 > 0:
            if stateInput[asix - 1][asiy][0] == 0 \
                    and stateInput[asix - 1][asiy][2] == 0\
                    and stateInput[asix - 1][asiy][6] == 0:
                actions[0] = 1
            else:
                actions[0] = 0
        else:
            actions[0] = 0
        #down
        if asix + 1 < self.hight:
            if stateInput[asix + 1][asiy][0] == 0 \
                    and stateInput[asix + 1][asiy][2] == 0 \
                    and stateInput[asix + 1][asiy][6] == 0:
                actions[1] = 1
            else:
                actions[1] = 0
        else:
            actions[1] = 0
        #left
        if asiy - 1 > 0:
            if stateInput[asix][asiy - 1][0] == 0 \
                    and stateInput[asix][asiy - 1][2] == 0\
                    and stateInput[asix][asiy - 1][6] == 0:
                actions[2] = 1
            else:
                actions[2] = 0
        else:
            actions[2] = 0
        # right
        if asiy + 1 < self.hight:
            if stateInput[asix][asiy + 1][0] == 0 \
                    and stateInput[asix][asiy + 1][2] == 0 \
                    and stateInput[asix][asiy + 1][6] == 0:
                actions[3] = 1
            else:
                actions[3] = 0
        else:
            actions[3] = 0
        # up left
        if asix - 1 > 0 and asiy - 1 > 0:
            if stateInput[asix - 1][asiy - 1][0] == 0 \
                    and stateInput[asix - 1][asiy - 1][2] == 0\
                    and stateInput[asix - 1][asiy - 1][6] == 0:
                actions[4] = 1
            else:
                actions[4] = 0
        else:
            actions[4] = 0
        #up right
        if asix - 1 > 0 and asiy + 1 < self.hight:
            if stateInput[asix - 1][asiy + 1][0] == 0 \
                    and stateInput[asix - 1][asiy + 1][2] == 0 \
                    and stateInput[asix - 1][asiy + 1][6] == 0:
                actions[5] = 1
            else:
                actions[5] = 0
        else:
            actions[5] = 0
        #down left
        if asix + 1 < self.hight and asiy - 1 > 0:
            if stateInput[asix + 1][asiy - 1][0] == 0 \
                    and stateInput[asix + 1][asiy - 1][2] == 0 \
                    and stateInput[asix + 1][asiy - 1][6] == 0:
                actions[6] = 1
            else:
                actions[6] = 0
        else:
            actions[6] = 0
        #down right
        if asix + 1 < self.hight and asiy + 1 < self.hight:
            if stateInput[asix + 1][asiy + 1][0] == 0 \
                    and stateInput[asix + 1][asiy + 1][2] == 0 \
                    and stateInput[asix + 1][asiy + 1][6] == 0:
                actions[7] = 1
            else:
                actions[7] = 0
        else:
            actions[7] = 0
        return actions