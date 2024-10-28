import torch as t
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

def dimRedByCnn(dim:int,filter:int,stride:int,padding:int)->int:
    newDim=(dim+2*padding-filter)//stride +1
    return newDim
class DQN(nn.Module):
    """
    the CNN model that is used to approximate the Q-function
    """

    def __init__(self, height:int, width:int, depth:int, n_actions:int, lr:float):
        """
         height:(int): height of the input
         width (int):  the width of the input 
         depth (int):  the number of images that are stacked
         n_actions (int): the number of action our agent can choose from
         lr (float) : the learning rate of the optimizer.
        """
        super(DQN, self).__init__()
        self.height=height
        self.width=width
        self.depth=depth
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        strideConv1=4
        padCV1=1
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=strideConv1, padding=padCV1)

        strideConv2=2
        padCV2=1
        secondChannel=64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=secondChannel, kernel_size=3, stride=strideConv2, padding=padCV2)
        strideConv3=1
        padCV3=0
        finalChannel=128
        self.conv3 = nn.Conv2d(in_channels=secondChannel, out_channels=finalChannel, kernel_size=3, stride=strideConv3, padding=padCV3)

        height=dimRedByCnn(height,8,strideConv1,padCV1)
        width=dimRedByCnn(width,8,strideConv1,padCV1)#conv1

        height=dimRedByCnn(height,3,strideConv2,padCV2)
        width=dimRedByCnn(width,3,strideConv2,padCV2)#conv2

        height=dimRedByCnn(height,3,strideConv3,padCV3)
        width=dimRedByCnn(width,3,strideConv3,padCV3)

    
        self._initialize_weights()
        self.fc1 = nn.Linear(height*width*finalChannel, 1024)
        self.fc2= nn.Linear(1024, n_actions)
        
    
        
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = t.device("cuda:0" if t.cuda.is_available() else 'cpu')
        
        if t.cuda.device_count() > 1:
            self.device1=t.device("cuda:1")
        else :
            self.device1=t.device("cpu")
        self.to(self.device)
    def forward(self, x:t.tensor)->t.tensor:
        x=x.reshape(-1,self.depth,self.height,self.width)
        x/=255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x       
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)