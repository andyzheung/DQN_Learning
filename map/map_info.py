import random
import numpy as np



class map:
    def __init__(self):
        self.width = 12
        self.hight = 12
        self.battlemap = []
        self.cacheMap = [[0 for col in range(self.width)] for row in range(self.hight)]
        self.PosionMap = [[0 for col in range(self.width)] for row in range(self.hight)]

    def init_battle_map(self):
        self.battlemap = np.array([
           [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, -3, -3, -3, -3, -3, -3, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, -3, -3, -3, 1, 1, -3, -3, -3, 1, 1],
            [1, 1, 1, 1, 1, -3, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, -3, 1, 1, -3, 1, 1, 1, 1],
            [1, 1, 1, -3, 1, 1, 1, 1, -3, 1, 1, 1],
            [1, 1, -3, 1, -3, -3, -3, -3, 1, -3, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -3, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, -3, 1, 1, 1, 1, 1, 1, -3, 1, 1, 1],
             [1, -3, 1, 1, 1, 1, 1, 1, -3, 1, 1, 1],
             [1, 1, -3, 1, 1, 1, 1, -3, 1, 1, 1, 1],
             [1, 1, 1, -3, 1, 1, -3, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, -3, -3, 1, 1, 1, 1, 1, 1],
             [1, -3, 1, 1, 1, 1, 1, 1, -3, 1, 1, 1],
             [1, -3, 1, 1, -3, -3, 1, 1, -3, 1, 1, 1],
             [1, -3, 1, 1, -3, -3, 1, 1, -3, 1, 1, 1],
             [1, 1, -3, -3, 1, 1, -3, -3, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, -3, -3, 1, 1, 1, 1, -3, -3, 1, 1, 1],
             [1, 1, 1, -3, 1, 1, -3, 1, 1, -3, 1, 1],
             [1, 1, 1, -3, 1, 1, -3, 1, 1, -3, 1, 1],
             [1, 1, -3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, -3, 1, 1, 1, 1, -3, 1, 1, -3, 1, 1],
             [1, -3, 1, 1, 1, 1, -3, 1, 1, -3, 1, 1],
             [1, -3, 1, 1, 1, 1, -3, 1, 1, 1, 1, 1],
             [1, 1, -3, -3, 1, 1, 1, -3, -3, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, -3, 1, 1, 1, -3, -3, -3, 1, 1, 1],
             [1, 1, -3, 1, 1, -3, 1, 1, 1, -3, 1, 1],
             [1, 1, -3, 1, 1, -3, 1, 1, 1, -3, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, -3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, -3, 1, 1, -3, 1, 1, 1, -3, 1, 1],
             [1, 1, -3, 1, 1, -3, 1, 1, 1, -3, 1, 1],
             [1, 1, -3, 1, 1, -3, 1, 1, 1, -3, 1, 1],
             [1, 1, 1, 1, 1, 1, -3, -3, -3, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

            [[1, -3, 1, 1, 1, -3, -3, 1, 1, 1, 1, 1],
             [-3, 1, 1, -3, 1, 1, 1, 1, -3, 1, 1, 1],
             [1, 1, -3, 1, 1, -3, -3, 1, 1, -3, 1, 1],
             [1, -3, 1, 1, -3, 1, 1, -3, 1, 1, 1, 1],
             [1, 1, 1, -3, 1, 1, 1, 1, -3, 1, 1, 1],
             [1, 1, -3, 1, 1, 1, 1, 1, 1, -3, 1, 1],
             [1, 1, -3, 1, 1, 1, 1, 1, 1, -3, 1, 1],
             [1, 1, 1, -3, 1, 1, 1, 1, -3, 1, 1, 1],
             [1, -3, 1, 1, -3, 1, 1, -3, 1, 1, -3, 1],
             [1, 1, -3, 1, 1, -3, -3, 1, 1, -3, 1, 1],
             [-3, 1, 1, -3, 1, 1, 1, 1, -3, 1, 1, -3],
             [1, -3, 1, 1, 1, -3, -3, 1, 1, 1, -3, 1]],

            [[1, 1, -3, 1, 1, -3, 1, 1, -3, 1, 1, 1],
             [1, -3, 1, 1, -3, 1, 1, -3, 1, 1, -3, 1],
             [-3, 1, 1, -3, 1, 1, -3, 1, 1, -3, 1, -3],
             [1, 1, -3, 1, 1, -3, 1, 1, -3, 1, 1, 1],
             [1, -3, 1, 1, 1, 1, 1, -3, 1, 1, -3, 1],
             [-3, 1, 1, -3, 1, 1, 1, 1, 1, 1, 1, -3],
             [1, 1, -3, 1, 1, 1, 1, 1, 1, -3, 1, 1],
             [1, -3, 1, 1, -3, 1, 1, 1, 1, 1, -3, 1],
             [-3, 1, 1, -3, 1, -3, 1, 1, -3, 1, 1, -3],
             [1, 1, -3, 1, 1, 1, -3, 1, 1, -3, 1, 1],
             [1, -3, 1, 1, -3, 1, 1, -3, 1, 1, -3, 1],
             [1, 1, -3, 1, 1, -3, 1, 1, -3, 1, 1, -3]],
                          ])
        self.battlemap.reshape(12,12,6)
        return 6

    def fillramdomplayer(self,i):
        self.cacheMap = self.battlemap[i,:,:].copy()
        s = []
        for j in range(len(self.cacheMap)):
            s.extend(self.cacheMap[j])
        self.stat = np.array(s)
        print("self.stat:",self.stat)
        x = self.stat.reshape(12,12)
        print("x",x)
        #找到等于1的位置
        newlist = []
        for k in range(len(self.stat)):
            if self.stat[k] == -3:
                newlist.append(k)
        nlist = range(0, 144)
        newlist0 = list(set(nlist).difference(set(newlist)))
        #从中随机抽取两个位置作为player
        # print("newlist",newlist)
        playerposlist = random.sample(newlist0, 2)
        print("playerposlist",playerposlist)
        # myplayer
        x, y = self.get_coordinate_from_index(playerposlist[0], self.width, self.width)
        self.cacheMap[x][y] = 0
        print("myplayer x y ", x, y)
        # enemy
        x, y = self.get_coordinate_from_index(playerposlist[1], self.width, self.width)
        self.cacheMap[x][y] = -1
        print("enmey x y ", x, y)
        reservelist = list(set(newlist0).difference(set(playerposlist)))
        return reservelist

    def calclooptimes(self,i):
        self.cacheMap = self.battlemap[i,:,:].copy()
        s = []
        for j in range(len(self.cacheMap)):
            s.extend(self.cacheMap[j])
        self.stat = np.array(s)
        print("self.stat:",self.stat)
        x = self.stat.reshape(12,12)
        print("x",x)
        #找到等于1的位置
        newlist = []
        for k in range(len(self.stat)):
            if self.stat[k] == -3:
                newlist.append(k)
        nlist = range(0, 144)
        newlist0 = list(set(nlist).difference(set(newlist)))
        result,resultlen = self.Combination(newlist0,2)
        return resultlen*2

    def fillplayerpositon(self,i,mapindex,flag):
        self.reset_cachemap()
        self.cacheMap = self.battlemap[mapindex,:,:].copy()
        s = []
        for j in range(len(self.cacheMap)):
            s.extend(self.cacheMap[j])
        self.stat = np.array(s)
        print("self.stat:",self.stat)
        x = self.stat.reshape(12,12)
        print("x",x)
        #找到等于1的位置
        newlist = []
        for k in range(len(self.stat)):
            if self.stat[k] == -3:
                newlist.append(k)
        nlist = range(0, 144)
        newlist0 = list(set(nlist).difference(set(newlist)))
        result,resultlen = self.Combination(newlist0,2)

        #从中随机抽取两个位置作为player
        # print("newlist",newlist)
        playerposlist = result.__getitem__(i)
        print("playerposlist",playerposlist)
        if flag == True:
            tmp = playerposlist[0]
            playerposlist[0] = playerposlist[i]
            playerposlist[i] = tmp
        # myplayer
        x, y = self.get_coordinate_from_index(playerposlist[1], self.width, self.width)
        self.cacheMap[x][y] = 0
        print("myplayer x y ", x, y)
        # enemy
        x, y = self.get_coordinate_from_index(playerposlist[0], self.width, self.width)
        self.cacheMap[x][y] = -1
        print("enmey x y ", x, y)
        reservelist = list(set(newlist0).difference(set(playerposlist)))
        return reservelist

    def Combination(self,list,k):
        n = len(list)
        result = []
        for i in range(n - k + 1):
            if k > 1:
                newL = list[i+1:]
                Comb, _ = self.Combination(newL,k - 1)
                for item in Comb:
                    item.insert(0,list[i])
                    result.append(item)
            else:
                result.append([list[i]])

        return result,len(result)


    def GeneratorPosion(self,traintimes):
        secure = 0
        Poisonlist = []
        secure += int(traintimes / 5)
        print("genertaton:",secure)
        if int(secure) > 0:
            if secure > 5:
                secure = secure % 5
            layer = secure
            print("posion layer:",layer)
            Poisonlist = self.fillposionaround(layer,self.hight,self.width)
            print("Poisonlist:",len(Poisonlist))
        return Poisonlist

    def GeneratorTool(self,reservelist,posionlist,i):
        if i % 5 != 0:
            return
        newlist = list(set(reservelist).difference(set(posionlist)))
        #随机抽取两位位置作为tool
        toollist = random.sample(newlist, 2)
        print("playerposlist", toollist)
        # myplayer
        x, y = self.get_coordinate_from_index(toollist[0], self.width, self.width)
        self.cacheMap[x][y] = 2
        print("attack x y ", x, y)
        # enemy
        x, y = self.get_coordinate_from_index(toollist[1], self.width, self.width)
        self.cacheMap[x][y] = 3
        print("defense x y ", x, y)



    def get_index_from_coordinate(self,x,y,hight):
        return hight*x+y

    def get_coordinate_from_index(self,index,width,hight):
        return index//width,index % hight

    def fillposionaround(self,layer,width,hight):
        self.posion_reset()
        walllist = []
        for i in range(0,layer):
            j=0
            j=j+i
            #上下边处理
            while (j < width - i):
                xIndex = j
                yIndex = i
                if xIndex < width and yIndex < hight:
                    self.PosionMap[xIndex][yIndex] = 4
                    arrayindex = self.get_index_from_coordinate(xIndex,yIndex,hight)
                    walllist.append(arrayindex)

                yIndex = width -i - 1
                if xIndex < width and yIndex < hight:
                    self.PosionMap[xIndex][yIndex] = 4
                    arrayindex = self.get_index_from_coordinate(xIndex,yIndex,hight)
                    walllist.append(arrayindex)
                j = j+1

            #左右边处理
            j = 1+i
            while(j < hight - i -1):
                #左边计算
                xIndex = i
                yIndex = j
                if xIndex < width and yIndex < hight:
                    self.PosionMap[xIndex][yIndex] = 4
                    arrayindex = self.get_index_from_coordinate(xIndex,yIndex,hight)
                    walllist.append(arrayindex)
                #右边计算
                xIndex = width - i -1
                if xIndex < width and yIndex < hight:
                    self.PosionMap[xIndex][yIndex] = 4
                    arrayindex = self.get_index_from_coordinate(xIndex,yIndex,hight)
                    walllist.append(arrayindex)
                j = j+1
        return walllist



    def fill_wall_around(self,layer,width,hight):
        walllist = []

        for i in range(0,layer):
            j=0
            j=j+i
            #上下边处理
            while (j < width - i):
                xIndex = j
                yIndex = i
                if xIndex < width and yIndex < hight:
                    self.cacheMap[xIndex][yIndex] = 1
                    arrayindex = self.get_index_from_coordinate(xIndex,yIndex,hight)
                    walllist.append(arrayindex)

                yIndex = width -i - 1
                if xIndex < width and yIndex < hight:
                    self.cacheMap[xIndex][yIndex] = 1
                    arrayindex = self.get_index_from_coordinate(xIndex,yIndex,hight)
                    walllist.append(arrayindex)
                j = j+1

            #左右边处理
            j = 1+i
            while(j < hight - i -1):
                #左边计算
                xIndex = i
                yIndex = j
                if xIndex < width and yIndex < hight:
                    self.cacheMap[xIndex][yIndex] = 1
                    arrayindex = self.get_index_from_coordinate(xIndex,yIndex,hight)
                    walllist.append(arrayindex)
                #右边计算
                xIndex = width - i -1
                if xIndex < width and yIndex < hight:
                    self.cacheMap[xIndex][yIndex] = 1
                    arrayindex = self.get_index_from_coordinate(xIndex,yIndex,hight)
                    walllist.append(arrayindex)
                j = j+1
        return walllist


    def GeneratorMap(self, walltotalnum, traintimes,width):
        #训练一定次数后，需要将周围变为墙壁
        secure = 0
        walllist = []
        secure += int(traintimes/200000)
        #将外围设置为墙壁
        fillwallaroundflag = True
        if int(secure) > 0:
            if secure > 5:
                secure = secure//5
            layer = secure
            #print("layer",layer)
            walllist = self.fill_wall_around(layer,width,width)
            if layer > 4:
                fillwallaroundflag = False
        # 墙壁的范围
        wallnum = random.randint(10,walltotalnum)
        # 从墙壁剩余集合中随机抽取wallnum个数字
        nlist = range(0,144)
        newlist0 = list(set(nlist).difference(set(walllist)))
        #print("newlist0",newlist0)
        if secure < 3:
            wallposlist = random.sample(newlist0,wallnum)
            #print("wallposlist",wallposlist)
            for i in range(len(wallposlist)):
                x,y = self.get_coordinate_from_index(wallposlist[i],width,width)
                self.cacheMap[x][y] = 1
                newlist = list(set(newlist0).difference(set(wallposlist)))
        else:
            newlist = newlist0
        if fillwallaroundflag == True:
            newlist1 = newlist
            if traintimes % 5 == 0:
               # 随机产生道具
               toolslist = random.sample(newlist, 2)
               # 攻击道具
               x, y = self.get_coordinate_from_index(toolslist[0], width, width)
               self.cacheMap[x][y] = 5
               # 防御道具
               x, y = self.get_coordinate_from_index(toolslist[1], width, width)
               self.cacheMap[x][y] = 6
               newlist = list(set(newlist1).difference(set(toolslist)))

        #print("newlist",newlist)
        playerposlist = random.sample(newlist,2)
        #print("playerposlist",playerposlist)
        #myplayer
        x,y = self.get_coordinate_from_index(playerposlist[0],width,width)
        self.cacheMap[x][y] = 3
        print("myplayer x y ",x, y)
        #enemy
        x, y = self.get_coordinate_from_index(playerposlist[1],width,width)
        self.cacheMap[x][y]= 4
        print("enmey x y ",x,y)

    def print_data1(self):
        for x in range(0,len(self.cacheMap)):
            for y in range(0,len(self.cacheMap[0])):
                print("%-9s" % (str(self.cacheMap[x][y]) + ""), end='')
            print("\n\n")

    def posion_reset(self):
        for x in range(0,len(self.PosionMap)):
            for y in range(0,len(self.PosionMap[0])):
                self.PosionMap[x][y] = 0

    def reset_cachemap(self):
        for x in range(0,len(self.cacheMap)):
            for y in range(0,len(self.cacheMap[0])):
                self.cacheMap[x][y] = 0
