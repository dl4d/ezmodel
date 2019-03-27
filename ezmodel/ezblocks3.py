from keras.layers import *


class Block:

    def __init__(self):
        self.block = None

    def define(self,block):
        self.block = block
        return self.new

    def new(self,*args,**kwargs):
        s = self.block
        for k,v in kwargs.items():
            tofind=k+"=?"
            replace=k+"="+str(v)
            s = s.replace(tofind,replace)
        a = s.splitlines()
        block=[]
        for l in a:
            if len(l)==0:
                continue;
            block.append(eval(l))
        return block
