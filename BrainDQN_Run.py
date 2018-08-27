import tensorflow as tf
import numpy as np
import random
from collections import deque
from Pos import Pos

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


class BrainDQNRun:

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

    def getAction(self,observation):

        print("getaction 0 ",observation[:, :, 0])
        print("getaction 1 ", observation[:, :, 1])
        print("getaction 2 ", observation[:, :, 2])
        print("getaction 3 ", observation[:, :, 3])
        print("getaction 4 ", observation[:, :, 4])
        print("getaction 5 ", observation[:, :, 5])
        print("getaction 6 ", observation[:, :, 6])

        print("currentState 0 ", self.currentState[:, :, 0])
        print("currentState 1 ", self.currentState[:, :, 1])
        print("currentState 2 ", self.currentState[:, :, 2])
        print("currentState 3 ", self.currentState[:, :, 3])
        print("currentState 4 ", self.currentState[:, :, 4])
        print("currentState 5 ", self.currentState[:, :, 5])
        print("currentState 6 ", self.currentState[:, :, 6])

        actionnew = self.getwallstate(self.currentState)
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState],
                                 self.a: [actionnew]})[0]
        print("Qvalue:",QValue)
        action = np.zeros(self.actions)
        action_index = 0
        action_index = np.argmax(QValue)
        print("Choose_action:", QValue)
        action[action_index] = 1
        move,observation_,done = self.get_next_observation(observation,action)
        return move,observation_,done


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


    def get_next_observation(self,observation,actions):
        player_index = (np.argwhere(observation[:, :, 1] == 1)).tolist()[0]
        asix_old, asiy_old = player_index[0], player_index[1]
        asix, asiy = player_index[0], player_index[1]

        done = False

        action_index = np.argwhere(actions == 1)
        action = action_index[0]

        if action == 0:  ##up
            asix -= 1
        elif action == 1:  ##down
            asix += 1
        elif action == 2:  ##left
            asiy -= 1
        elif action == 3:  ##right
            asiy += 1
        elif action == 4:  ##up left
            asix -= 1
            asiy -= 1
        elif action == 5:  ##up right
            asix -= 1
            asiy += 1
        elif action == 6:  ##down left
            asix += 1
            asiy -= 1
        elif action == 7:  ##down right
            asix += 1
            asiy += 1
        elif action == 8:  ##done
            done = True

        observation[player_index[0]][player_index[1]][6] = 1

        em_index = (np.argwhere(observation[:, :, 2] == 1)).tolist()[0]
        ex, ey = em_index[0], em_index[1]

        move = []

        if (asix < 0 or asix >= self.hight
            or asiy < 0 or asiy >= self.hight
            or observation[asix][asiy][0] == 1 #墙壁
            or observation[asix][asiy][6] == 1 #已经走过
            or (ex == asix and ey == asiy)):
            print("Path is invalid")
            return move,observation,True

        if done != True:
            observation[asix_old][asiy_old][1] = 0 #恢复为0
            observation[asix][asiy][1] = 1         #设置为1
            move = Pos(0,0)
            move.x = asix
            move.y = asiy
        return move,observation,done



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