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
        blocks=[]
        for l in a:
            if len(l)==0:
                continue;
            blocks.append(eval(l))
        return blocks


class Graph:

    def __init__(self,data):

        if type(data) is list:
            print("Multi input Not implemented yet")
            return

        if len(data.shape)<4:
            print("Table input Not implemented yet")
            return

        self.network = Input(data.shape[1:])
        self.levels = [(0,0,None,Input(data.shape[1:]))] #(id,level,parent,content)

        print(self.levels)

    def __call__(self,level="",object=None):

        id = len(self.levels)
        parent = self.levels[id-1][0]
        print(len(level))
        print(self.levels[parent][1])
        if len(level)==self.levels[parent][1]:
            self.levels.append((id,len(level),parent,object))
        print(self.levels)

        # if type(object) is list:
        #     for layer in object:
        #         self.network = layer (self.network)
        #
        # if len(level) in self.levels:
        #     self.levels[len(level)].append(self.network)
        # else:
        #     self.levels[len(level)] = [self.network]
        #
        # print(self.network)
        # print(self.levels)
        #



# class eznet:
#     def __init__(self,data=None):
#         self.data = data
#         self.network = None
#         self.input = []
#         self.output = []
#
#         if len(data.X.shape)==4:
#             input = Input(data.X.shape[1:])
#         else:
#             print('ezmodel.eznet().__init__(): Not implemented yet for tables ')
#             return
#
#         self.network = input
#         self.input.append(input)
#
#
#     def add(self,object):
#
#         if type(object) is list:
#             previous = self.network
#             for layer in object:
#                 self.network = (layer) (previous)
#
#         else:
#             previous = self.network
#             self.network = (object) (previous)
#
#     # def output(self,object):
