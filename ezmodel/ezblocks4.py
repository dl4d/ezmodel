from keras.layers import *


class Block:

    def __init__(self):
        self.block = None
        self.blocks = None

    def define(self,block):
        self.block = block
        return self.new

    def __call__(self,tensor):
        tensor = self.blocks[0] (tensor)
        for m in self.blocks[1:]:
            tensor = m (tensor)
        return tensor


    def new(self,*args,**kwargs):
        s = self.block
        s = s.strip()
        for k,v in kwargs.items():
            tofind=k+"=?"
            replace=k+"="+str(v)
            s = s.replace(tofind,replace)
        a = s.splitlines()
        blocks=[]
        for l in a:
            if len(l)==0:
                continue;
            blocks.append(eval(l))
        self.blocks=blocks
        return self
