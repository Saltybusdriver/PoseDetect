class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = out + residual
        out = self.relu(out)
        return out
    
class CenterBlocks(nn.Module):
    def __init__(self,in_channels):
        super(CenterBlocks, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = out + residual
        out = self.relu(out)
        return out
    
class SkipResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = out + residual
        out = self.relu(out)
        return out


class HourglassModule(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(HourglassModule, self).__init__()
        self.depth = depth
        self.downsample = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.residual_blocks = nn.ModuleList([ResidualBlock(in_channels, out_channels) for _ in range(depth)])
        self.skip_residual_blocks = nn.ModuleList([ResidualBlock(in_channels, out_channels) for _ in range(depth)])
        self.match_channels = nn.Conv2d(16, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.Skip = None
        self.relu = nn.LeakyReLU(inplace=True)
        self.bottleneck1 = nn.Conv2d(256, 256, kernel_size=1)
        self.bottleneck2 = nn.Conv2d(256, 256, kernel_size=1)
        self.bottleneck3 = nn.Conv2d(256, 256, kernel_size=1)
        self.CenterBlocks= CenterBlocks(in_channels)
    def forward(self, x,prev_features):
        self.Skip=[]
        if(prev_features is not None):
            x = prev_features
        for i in range(self.depth):
            x = self.residual_blocks[i](x)
            x = self.residual_blocks[i](x)
            x = self.residual_blocks[i](x)
            self.Skip.append(x)
            self.Skip[i]=self.skip_residual_blocks[i](self.Skip[i])
            x = self.downsample(x)
       
        x=self.CenterBlocks(x)
        x=self.CenterBlocks(x)
        x=self.CenterBlocks(x)
        for i in range(self.depth):
            x = self.upsample(x)
            x =x+ self.Skip[(self.depth-1)-i]
            x = self.residual_blocks[i](x)
            x = self.residual_blocks[i](x)
            x = self.residual_blocks[i](x)
        if(prev_features is not None):
            x=x+prev_features
        return x


class StackedHourglassFCN(nn.Module):
    def __init__(self, num_stacks, num_joints, depth):
        super(StackedHourglassFCN, self).__init__()
        self.num_stacks = num_stacks
        self.num_joints = num_joints
        self.depth = depth
        self.input_conv = nn.Conv2d(1, 256, kernel_size=7,stride=1,padding=3)
        self.input_convNetwork = nn.Conv2d(256, 256, kernel_size=7,stride=1,padding=3)
        self.input_bn = nn.BatchNorm2d(256)
        self.input_relu = nn.ReLU()
        self.hourglass_stacks = nn.ModuleList([HourglassModule(128, 128, depth) for _ in range(num_stacks)])
        self.outputc=nn.Conv2d(128,num_joints,kernel_size=1)
        self.outputc2=nn.Conv2d(num_joints,num_joints,kernel_size=1)
        self.outputc3=nn.Conv2d(num_joints,num_joints,kernel_size=1)
        self.nextmod=nn.Conv2d(256,128,kernel_size=1)
        self.nextmod2=nn.Conv2d(128,256,kernel_size=1)
        self.heatnext=nn.Conv2d(num_joints,256,kernel_size=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        outputs = []
        prev_map=None
        oinput=None
        losss = nn.MSELoss()
        for i in range(self.num_stacks):
            if(prev_map is  not None):
                oinput=x
            if(prev_map is not None):
                x=self.input_convNetwork(x)
            else:
                x = self.input_conv(x)
            x = self.input_bn(x)
            x = self.input_relu(x)
            x = self.hourglass_stacks[i](x,prev_map)
            x=self.nextmod(x)
            x = self.relu(x)
            heatmap=x
            heatmap=self.outputc(heatmap)
            heatmap=self.relu(heatmap)
            heatmap=self.outputc2(heatmap)
            heatmap=self.relu(heatmap)
            heatmap=self.outputc3(heatmap)
            output=heatmap
            
            if(i != self.num_stacks-1):
                heatmap=self.heatnext(heatmap)
                x=self.nextmod2(x)
                x=self.relu(x)
                if(oinput is not None):
                    prev_map=heatmap+x+oinput
                else:
                    prev_map=heatmap+x
            outputs.append(output)
        return torch.stack(outputs, dim=1)
        


         
#model=StackedHourglassFCN(8,16,4)
#model.to(dtype=torch.bfloat16, device=dev)