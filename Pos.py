def loadJson(dict):
    return Pos(dict["x"],dict["y"])

class Pos:
    def __init__(self,x,y):
        self.x = x
        self.y = y