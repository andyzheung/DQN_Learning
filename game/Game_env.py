
import numpy as np
import pandas as pd
import logging



class Game:
    def __init__(self):
        super(Game,self).__init__()

        '''
        u --up
        d --down
        l --left
        r --right
        a --upper left corner
        x --upper right corner
        s --lower left conrner 左下角
        w --lower right corner 
        e --end action
        '''
        '''
        M位置为0，敌人位置 - 1，走过位置 - 2，墙位置 - 3，可走路径1，攻击2，防具
        '''
        self.action_space = ['u','d','l','r','a','x','s','w','e']
        self.n_actions = len(self.action_space)
        self.n_features = 146
        self.title('Jedi survival')
        self.stat = np.array([])
        self.binary_env = np.array([],dtype=float)
        self.reward = 0
        self.x_len = 0
        self.y_len = 0
        self.use_or_no = False
        self._build_game_env()

    @staticmethod
    def title(strr):
        return strr

    def _build_game_env(self):
        pass

    @staticmethod
    def get_frame(map_array):
        return pd.DataFrame(map_array)

    def get_start_point(self):
        return self.stat

    '''
    M位置为0，敌人位置 - 1，走过位置 - 2，墙位置 - 3，可走路径1，攻击2，防具3
    #六个平面，
    第0维  空地和墙 
    第1维 我方选手 
    第2维 对方选手 
    第3维  毒气 
    第4维  防御 
    第5维  攻击
    ## 当前位置0，敌人位置-1，已经走过位置-2，墙为-3 空地为1，攻击为2，防御为3,毒气为4
    '''
    def get_wall_layer(self,map_info):
        a = np.zeros(shape = [self.x_len,self.y_len],dtype= float)
        b = np.array(map_info)
        v = np.argwhere(b == -3)
        for idx in v:
            a[idx[0]][idx[1]] = 1
        return a

    def get_player_layer(self,map_info):
        a = np.zeros(shape = [self.x_len,self.y_len],dtype= float)
        b = np.array(map_info)
        v = np.argwhere(b == 0)
        for idx in v:
            a[idx[0]][idx[1]] = 1
        return a

    def get_enemy_layer(self,map_info):
        a = np.zeros(shape = [self.x_len,self.y_len],dtype= float)
        b = np.array(map_info)
        v = np.argwhere(b == -1)
        for idx in v:
            a[idx[0]][idx[1]] = 1
        return a

    def get_posion_layer(self,map_info):
        a = np.zeros(shape = [self.x_len,self.y_len],dtype= float)
        b = np.array(map_info)
        v = np.argwhere(b == 4)
        for idx in v:
            a[idx[0]][idx[1]] = 1
        return a

    def get_defense_layer(self,map_info):
        a = np.zeros(shape = [self.x_len,self.y_len],dtype= float)
        b = np.array(map_info)
        v = np.argwhere(b == 3)
        for idx in v:
            a[idx[0]][idx[1]] = 1
        return a

    def get_attack_layer(self,map_info):
        a = np.zeros(shape = [self.x_len,self.y_len],dtype= float)
        b = np.array(map_info)
        v = np.argwhere(b == 2)
        for idx in v:
            a[idx[0]][idx[1]] = 1
        return a

    def binary_env_reset(self,map_info,posion_info):
        self.y_len = len(map_info)
        self.x_len = len(map_info[0])
        self.reward = 0
        self.attach = 0
        self.posionflg = False
        s = []
        for i in range(len(map_info)):
            s.extend(map_info[i])
        self.stat = np.array(s)
        self.stat = np.append(self.stat, [0, 0])
        a = self.get_wall_layer(map_info)   #第0维  空地和墙
        b = self.get_player_layer(map_info) #第1维 我方选手
        c = self.get_enemy_layer(map_info)  #第2维 对方选手
        d = self.get_posion_layer(posion_info) #第3维  毒气
        e = self.get_defense_layer(map_info)  #第4维  防御
        f = self.get_attack_layer(map_info)   #第5维  攻击
        g = np.zeros(shape = [self.x_len,self.y_len],dtype= float) #第6维 走过的路
        self.binary_env = np.zeros(shape = [self.x_len,self.y_len,7],dtype = float)
        self.binary_env[:, :, 0] = a
        self.binary_env[:, :, 1] = b
        self.binary_env[:, :, 2] = c
        self.binary_env[:, :, 3] = d
        self.binary_env[:, :, 4] = e
        self.binary_env[:, :, 5] = f
        self.binary_env[:,:,6] = g
        return self.binary_env

        #六个平面，0 第0维  空地和墙 第1维 我方选手 第2维 对方选手 第3维  毒气 第4维  防御 第5维  攻击
        ## 当前位置0，敌人位置-1，已经走过位置-2，墙为-3 空地为1，攻击为2，防御为3
    def binary_step(self, actionnew):
        player_index = (np.argwhere(self.binary_env[:,:,1] == 1)).tolist()[0]
        asix_old, asiy_old = player_index[0], player_index[1]
        asix, asiy = player_index[0], player_index[1]

        done = False
        reward = 0

        action_index = np.argwhere(actionnew == 1)
        action = action_index[0]
        print("action", action)
        print("player:",asix,asiy)


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

        em_index = (np.argwhere(self.binary_env[:,:,2] == 1)).tolist()[0]
        ex, ey = em_index[0], em_index[1]
        times = self.judgeattacktimes(ex,ey)
        print("enemy:x,y: my newpos,attacktimes:",ex, ey,asix, asiy,times)
        if action == 8:
            if self.binary_env[asix][asiy][3] == 1:
                self.posionflg = True
            if self.attach == 0:
                reward = - 1
            else:
                if self.attach == times:
                     reward += 0.1 * self.attach
                print("action 8,attach",self.attach)
                if asix -1 < 0 or asix + 1 >= self.x_len:
                    reward += 0.2
                    print("up or down out of map")
                    logging.info("up or down out of map")
                    #上面墙
                else:
                    if 1 == self.binary_env[asix -1][asiy][0]:
                        reward += 0.2
                        print("stop up wall")
                    #下边墙
                    if 1 == self.binary_env[asix +1][asiy][0]:
                        reward += 0.2
                        print("stop down wall")
                        logging.info("stop down wall")
                if asiy -1 < 0 or asiy + 1 >= self.y_len:
                    reward += 0.2
                    print("left or right wall")
                    logging.info("left or right wall")
                else:
                    #左面墙
                    if 1 == self.binary_env[asix][asiy-1][0]:
                        reward += 0.2
                        print("stop down wall")
                        logging.info("stop down wall")
                    #右边墙
                    if 1 == self.binary_env[asix][asiy+1][0]:
                        reward += 0.2
            if self.posionflg == True:
                reward -= 0.2
            print("action 8:reward",reward)
            logging.info("action 8:reward %f",reward)
        elif asix < 0 or asix >= self.x_len or asiy < 0 or asiy >= self.y_len:
            reward = -1
            done = True
            print("out of map")
            logging.info("out of map")
        #墙壁
        elif 1 == self.binary_env[asix][asiy][0]:
            reward = -1
            done = True
            print("on the wall")
        #对手
        elif 1 == self.binary_env[asix][asiy][2]:
            reward = -1
            done = True
            print("on the enemy")
        #已经走过
        elif 1 == self.binary_env[asix][asiy][6]:
            reward = -1
            done = True
            print("has already go")
            logging.info("has already go")
        elif asix == ex - 1 and asiy == ey:
            reward = 1
            self.attach += 1
            print("on the enemy up")
            logging.info("on the enemy up")
            done = False
        elif asix == ex + 1 and asiy == ey:
            reward = 1
            self.attach += 1
            print("on the enemy down")
            logging.info("on the enemy down")
            done = False
        elif asix == ex and asiy == ey - 1:
            reward = 1
            self.attach += 1
            print("on the enemy left")
            logging.info("on the enemy left")
            done = False
        elif asix == ex and asiy == ey + 1:
            reward = 1
            self.attach += 1
            print("on the enemy right")
            logging.info("on the enemy right")
            done = False
        elif 1 == self.binary_env[asix][asiy][4] \
            or 1 == self.binary_env[asix][asiy][5]:
            reward = 0.3
            done = False
            print("get tools")
        elif self.binary_env[asix][asiy][3] == 1:
            self.posionflg = True
            reward = 0
        else:
            reward = 0
        

        if done != True:
            #已经走过的路径
            self.binary_env[player_index[0]][player_index[1]][6] = 1
            self.binary_env[asix][asiy][6] = 1  #已经走过
            self.binary_env[asix_old][asiy_old][1] = 0 #恢复为0
            self.binary_env[asix][asiy][1] = 1         #新位置设置为1

        return self.binary_env.copy(), reward, done

    def judgeattacktimes(self,ex,ey):
        times = 0
        if ex -1 > 0 and self.binary_env[ex-1][ey][0] == 0:
            times += 1
        if ex + 1< self.y_len and self.binary_env[ex+1][ey][0] == 0:
            times += 1
        if ey - 1 > 0 and self.binary_env[ex][ey - 1][0] == 0:
            times += 1
        if ey + 1< self.y_len and self.binary_env[ex][ey + 1][0] == 0:
            times += 1
        return times






