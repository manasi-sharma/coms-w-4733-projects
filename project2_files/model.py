import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        # TODO
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(384, 128, 3, padding=1) #256
        self.conv7 = nn.Conv2d(192, 64, 3, padding=1) #128
        self.conv8 = nn.Conv2d(96, 32, 3, padding=1) #64
        self.conv9 = nn.Conv2d(48, 16, 3, padding=1) #32
        self.conv10 = nn.Conv2d(16, 6, 1)

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        # TODO
        
        # left-side
        x= F.relu(self.conv1(x))
        x_c1= x.clone()
        x = F.max_pool2d(x, 2)
        
        x= F.relu(self.conv2(x))
        x_c2= x.clone()
        x = F.max_pool2d(x, (2, 2))
        
        x= F.relu(self.conv3(x))
        x_c3= x.clone()
        x = F.max_pool2d(x, (2, 2))
        
        x= F.relu(self.conv4(x))
        x_c4= x.clone()
        x = F.max_pool2d(x, (2, 2))
        
        # right-side
        x_c5= F.relu(self.conv5(x))
        x_c5= F.interpolate(x_c5, scale_factor=2)
        x = torch.cat((x_c4, x_c5), dim=1)
        
        x_c6= F.relu(self.conv6(x))
        x_c6= F.interpolate(x_c6, scale_factor=2)
        x = torch.cat((x_c3, x_c6), dim=1)
        
        x_c7= F.relu(self.conv7(x))
        x_c7= F.interpolate(x_c7, scale_factor=2)
        x = torch.cat((x_c2, x_c7), dim=1)
        
        x_c8= F.relu(self.conv8(x))
        x_c8= F.interpolate(x_c8, scale_factor=2)
        x = torch.cat((x_c1, x_c8), dim=1)
        
        # end-segment
        x= F.relu(self.conv9(x))
        x= F.relu(self.conv10(x))        
        
        output = x
        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
