import numpy as np
from Pos import Pos


class AccuracyJudges:
    def __init__(self):
        self.cacheMap = []
        self.hight = 12
        self.width = 12
        self.battlemap = []

    def JudgeIfTheBestPractise(self,observation,moves,Posion):

        em_index = (np.argwhere(observation[:, :, 2] == 1)).tolist()[0]
        ex, ey = em_index[0], em_index[1]
        times = 0

        if ex -1 > 0 and observation[ex-1][ey][0] == 0:
            times += 1
        if ex + 1< self.hight and observation[ex+1][ey][0] == 0:
            times += 1
        if ey - 1 > 0 and observation[ex][ey - 1][0] == 0:
            times += 1
        if ey + 1< self.hight and observation[ex][ey + 1][0] == 0:
            times += 1

        attacktimes = 0

        for i in range(0,len(moves)):
            if moves[i].x == ex - 1 and moves[i].y == ey:
                attacktimes += 1
            if moves[i].x == ex + 1 and moves[i].y == ey:
                attacktimes += 1
            if moves[i].x == ex and moves[i].y == ey - 1:
                attacktimes += 1
            if moves[i].x == ex and moves[i].y == ey - 1:
                attacktimes += 1

        if (attacktimes == times):
            print("attack best!")

        if len(moves) == 0:
            return

        asixend = moves[len(moves) - 1].x
        asiyend = moves[len(moves) - 1].y

        wallnum = 0

        if asixend - 1 < 0 or asixend + 1 >= self.hight:
            wallnum += 1
        if asiyend - 1 < 0 or asiyend + 1 >= self.hight:
            wallnum += 1
        if asixend - 1 > 0 and observation[asixend - 1][asiyend][0] == 1:
            wallnum += 1
        if asixend + 1 < self.hight and observation[asixend + 1][asiyend][0] == 1:
            wallnum += 1
        if asiyend - 1 > 0 and observation[asixend][asiyend - 1][0] == 1:
            wallnum += 1
        if asiyend + 1 < self.hight and observation[asixend][asiyend+1][0] == 1:
            wallnum += 1

        if wallnum >=2:
            print("get wall good:",wallnum)

        print("get attacktimes, wallnum:",attacktimes,wallnum)
