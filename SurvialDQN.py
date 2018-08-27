# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

from game.Game_env import Game
from BrainDQN_Nature import BrainDQN
from map.map_info import map
import logging
from BrainDQN_Run import BrainDQNRun
from Pos import Pos
from Accuracy import AccuracyJudges

def playSurvival():
    logging.basicConfig(filename='logs/myplayer.log', level=logging.INFO)
    logging.info('Started')
    # 9个方向
    action_space = ['u', 'd', 'l', 'r', 'a', 'x', 's', 'w', 'e']
    n_actions = len(action_space)
    train = BrainDQN(n_actions)
    # 初始化随机地图类
    genermap = map()
    TrainGame = Game()
    for i in range(6000000):
        loop = 0
        loop += int(i / 1000000)
        print("loop:", loop)
        logging.info('loop %d,i %d', loop, i)
        # 初始化随机地图信息
        genermap.init_battle_map()
        reservelist = genermap.fillramdomplayer(loop)
        print("Posion start")
        #产生毒气
        posionlist = genermap.GeneratorPosion(i)
        print("Posion end")
        #产生道具
        genermap.GeneratorTool(reservelist,posionlist,i)
        # 初始化Game环境
        TrainGame.binary_env_reset(genermap.cacheMap,genermap.PosionMap)
        train.setInitState(TrainGame.binary_env)
        # 循环训练1W次，换一张地图
        for episode in range(2000):
            # 随机取一个方向
            action = train.getAction()
            # 计算该方向的reward
            nextObservation, reward, terminal = TrainGame.binary_step(action)
            # 设置到训练集
            train.setPerception(nextObservation, action, reward, terminal)
            if terminal == True:
                TrainGame.binary_env_reset(genermap.cacheMap,genermap.PosionMap)
                train.setInitState(TrainGame.binary_env)
                break


def TestTrain():
    action_space = ['u', 'd', 'l', 'r', 'a', 'x', 's', 'w', 'e']
    n_actions = len(action_space)

    # 初始化随机地图类
    genermap = map()
    TrainGame = Game()

    train = BrainDQNRun(n_actions)

    Judges = AccuracyJudges()
    for i in range(100):
        loop = 0
        loop += int(i / 1000000)
        genermap.init_battle_map()
        poisonlist = genermap.GeneratorPosion(i)
        genermap.fillramdomplayer(loop)
        # 初始化Game环境
        TrainGame.binary_env_reset(genermap.cacheMap, genermap.PosionMap)
        train.setInitState(TrainGame.binary_env)
        done = False
        moves = []
        observation = TrainGame.binary_env
        while done != True:
            movetmp,observation_,done = train.getAction(observation)
            observation = observation_
            if done != True:
                moves.append(movetmp)
        print("moves:",len(moves))
        for i in range(len(moves)):
            print(" ",moves[i].x,moves[i].y)
        Judges.JudgeIfTheBestPractise(TrainGame.binary_env,moves,genermap.PosionMap)


def PlaySurvalNew():
    logging.basicConfig(filename='logs/myplayer.log', level=logging.INFO)
    logging.info('Started')
    # 9个方向
    action_space = ['u', 'd', 'l', 'r', 'a', 'x', 's', 'w', 'e']
    n_actions = len(action_space)
    train = BrainDQN(n_actions)
    # 初始化随机地图类
    genermap = map()
    TrainGame = Game()
    mapnum = genermap.init_battle_map()
    for i in range(mapnum):
        #计算每张地图循环多少次
        looptimes = genermap.calclooptimes(i)
        logging.info('loop %d,i %d', looptimes, i)
        for j in range(looptimes):
            flag = False
            index = j
            if j >= looptimes/2:
                flag = True
                index = j - looptimes
            # 初始化随机地图信息
            reservelist = genermap.fillplayerpositon(index,i,flag)
            # 产生毒气
            posionlist = genermap.GeneratorPosion(i)
            # 产生道具
            genermap.GeneratorTool(reservelist, posionlist, i)
            # 初始化Game环境
            TrainGame.binary_env_reset(genermap.cacheMap, genermap.PosionMap)
            train.setInitState(TrainGame.binary_env)
            # 循环训练1W次，换一张地图
            for episode in range(5000):
                # 随机取一个方向
                action = train.getAction()
                # 计算该方向的reward
                nextObservation, reward, terminal = TrainGame.binary_step(action)
                # 设置到训练集
                train.setPerception(nextObservation, action, reward, terminal)
                if terminal == True:
                    TrainGame.binary_env_reset(genermap.cacheMap, genermap.PosionMap)
                    train.setInitState(TrainGame.binary_env)
                    #break

def main():
    #playSurvival()
    PlaySurvalNew()
    #TestTrain()

if __name__ == '__main__':
    main()
