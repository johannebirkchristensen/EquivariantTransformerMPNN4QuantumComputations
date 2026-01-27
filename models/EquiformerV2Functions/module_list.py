import torch

#This is a class for adding comments to ModuleLists
#lets say we define layers as a 3 layer basic NN, ml = ModuleListInfo("My simple 3-layer network", layers) then print(ml.info_str) 
# then we get "My simple 3-layer network"... smart for keeping track of complex ModuleLists 
class ModuleListInfo(torch.nn.ModuleList):
    def __init__(self, info_str, modules=None):
        super().__init__(modules)
        self.info_str = str(info_str)
    

    def __repr__(self): 
        return self.info_str