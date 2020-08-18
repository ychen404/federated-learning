class Test():
    def __init__(self, val):
        self.v = val
    
    def printVal(self):
        print(self.v)
    
    def MoreFunc(self):
        self.printVal()



T = Test(10)
T.printVal()
T.MoreFunc()
