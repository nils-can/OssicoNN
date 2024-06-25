import torch.nn as nn
import torch
# from ray import train, tune


class PrintSizeLayer(nn.Module):
    def __init__(self):
        super(PrintSizeLayer, self).__init__()

    def forward(self, x):
        # Print the size of the input tensor
        print(f"Input size: {x.size()}")
        return x
    
class DynamicLinear(nn.Module):
    def __init__(self, output_size):
        super(DynamicLinear, self).__init__()
        self.output_size = output_size
        self.fc = None  # We'll initialize this in the first forward pass

    def forward(self, x):
        if self.fc is None:
            input_size = x.size(1)
            self.fc = nn.Linear(input_size, self.output_size).cuda()
        x = self.fc(x)
        return x

# Conditioning Neural Network 
class CondNet_girafe(nn.Module):
    def __init__(self):
        super().__init__()
        #Conditioning level
        self.resolution_levels = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, 256, 64, 2),
                          nn.ReLU(),
                          nn.MaxPool1d(4),
                          nn.Conv1d(256, 256, 16, 2)),

            nn.Sequential(nn.ReLU(),
                          nn.MaxPool1d(1),
                          nn.Conv1d(256, 256, 128, 1)),

            nn.Sequential(nn.ReLU(),
                          nn.Conv1d(256, 16, 4, 2)),

            nn.Sequential(nn.Flatten(),
                          #nn.Linear(29120, 512),
                          DynamicLinear(64).cuda(),
                          nn.ReLU(),
                          nn.Dropout(0.33),
                          nn.Linear(64, 128),
                          nn.ReLU(),
                          nn.Dropout(0.33),
                          nn.Linear(128, 256),
                          nn.ReLU(),
                          nn.Dropout(0.33),
                          nn.Linear(256, 32))])
        
        # limk between the conditioning and the conditional neural network
        self.flatten_after_conv = nn.ModuleList([
            nn.Sequential(nn.ReLU(),
                          nn.MaxPool1d(4),
                          nn.Flatten(),
                          DynamicLinear(256).cuda(),
                          #nn.Linear(15008,64),
                          nn.ReLU(),
                          nn.Linear(256,256)),
            nn.Sequential(nn.ReLU(),
                          nn.MaxPool1d(1),
                          nn.Flatten(),
                          #nn.Linear(9856,64),
                          DynamicLinear(16).cuda(),
                          nn.ReLU(),
                          nn.Linear(16,32)),
            nn.Sequential(nn.ReLU(),
                          nn.MaxPool1d(1),
                          nn.Flatten(),
                          #nn.Linear(9664,64),
                          DynamicLinear(512).cuda(),
                          nn.ReLU(),
                          nn.Linear(512,256))])
        

    def forward(self, c):
        y = [c]
        for m in self.resolution_levels:
            y.append(m(y[-1]))
        for i in range(3):
            y[i + 1] = self.flatten_after_conv[i](y[i + 1])
        return y[1:]