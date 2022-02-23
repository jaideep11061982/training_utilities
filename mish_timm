import timm
import torch.nn.functional as F
grp_length=1
class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)
            
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

from timm.models.layers.activations import *

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    
    
    
class custom_model(nn.Module):
    def __init__(self,n=NUM_CLASSES,name=None):
        super().__init__()
        print('x')  
        #self.model=nn.Sequential((*list(get_model(name).children())[:-2]))
        self.model=get_model(name)
        #self.model.classifier=nn.Linear(1280,n)
        #w = self.model[0].weight.sum(1).unsqueeze(1)
        #print(self.model[0])
        #self.model[0] = Conv2dSame(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        #self.model[0].weight = nn.Parameter(w)
        #print(list(self.model.children())[-5].out_channels ) 
        #print(self.model)
        #print()
        #nc=list(get_model(name).children())[-5].out_channels
        #nc=list(get_model(name).children())[-3].out_channels
        #print(nc)
        #self.head= nn.Sequential( AdaptiveConcatPool2d(),Flatten(),nn.BatchNorm1d(2*nc ) ,
        #                         nn.Linear(2*nc ,512), HardSwish() ,nn.Dropout(0.3),nn.Linear(512,n))
        self.group_length=grp_length
    
     
    def forward(self,x):
        if self.group_length !=1:

          n=len(x)
          shape=x[0].shape
          #x=torch.stack(x,0)
          #print(torch.stack(x,1).shape)
          x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
          
          #print(shape,x.shape,n)
          #n,bs=x.shape[0],x.shape[1]
          #x=x.view(x.shape[1]*n,x.shape[2],x.shape[3],x.shape[4])
          #x=x.view(x.shape[0]*n,x.shape[])
          #print(x.size())
          x=self.model(x)
          shape = x.shape
          #print(x.size())
          #print(x.size(),x.view(-1,n,shape[1],shape[2],shape[3]).size(),
          #      x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).size())
          x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
            .view(-1,shape[1],shape[2]*n,shape[3])
          #x: bs x C x N*4 x 4
          #print(x.size())
        else:
          x=self.model(x)
        #x=self.head(x)
        return x
         
    ''' 
    def forward(self,x):
        
        #print(x.size())
        x=self.model(x)
         
        #print(x.size())
        #print(x.size(),x.view(-1,n,shape[1],shape[2],shape[3]).size(),
        #      x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).size())
        
        x=self.head(x)
        return x
    '''
         
#net=custom_model(name='tf_mobilenetv3_large_100')
net=custom_model(name='resnest26d')
