#from sklearn.preprocessing import LabelEncoder
#from IPython.display import clear_output
import Model
import heatmap
import Loss
import Loader
import sys
import getopt
import argparse

parser = argparse.ArgumentParser(description='Description of your script.')

parser.add_argument('-b', '--batch', type=int, default=1, help='Sets the batch size.')
parser.add_argument('-s', '--stack', type=int, default=8, help='Sets the amount of hourglass modules stacked.')
parser.add_argument('-d', '--depth', type=int, default=4, help='Sets the amount of residual and downscale pairs on a single side in the network.')
parser.add_argument('-S', '--size', type=int, default=96, help='Sets the resolution of the modified image, only accepts a single integer since the height and width dimensions are the same.')
parser.add_argument('-mb', '--maxbatch', type=int, default=None, help='Sets the maximum number of batches to use during training.')
parser.add_argument('-e', '--epoch', type=int, default=300, help='Sets the amount of epochs during training.')
parser.add_argument('-nd', '--newdataset', type=boolean, default=False, help='Tells the dataloader to do additional checks on the dataset since its new.')
args = parser.parse_args()
arg_batch=args.batch
arg_stack=args.stack
arg_depth=args.depth
arg_size=args.size
arg_epoch=args.epoch
arg_maxbatch=args.maxbatch
arg_newdata=args.newdataset
num_joints = 16
num_body_parts = 32
Loader=loader



if(arg_newdata==True):
    Loader.dataframe_editing()
else:
    Loader.LoadCleanedCSV()



Model=model.StackedHourglassFCN(arg_stack,num_joints,arg_depth)
Model.to(dtype=torch.bfloat16, device=dev)
criterion = loss.lossLossCalculation()
weight_dec = 1e-5
optimizer = torch.optim.Adam(Model.parameters(),lr=0.001)

_train_loader=Loader.train_loader



def train2(model, dev, _train_loader, optimizer, num_epochs=10000, batch_size=BATCH_SIZE):
    model.train()
    id=0

    height, width = New_width, New_height
    num_joints=16
    for epoch in range(10000):
        print("Loading Epoch...")
        for data, target in _train_loader:
            #####unsquueze so i have channel dimension
            data = data.unsqueeze(1)
            target=target.reshape(BATCH_SIZE,num_joints,2)  ## reshape target so that pairs of xy are next to each other..
            joint_mapss = torch.zeros(BATCH_SIZE, num_joints, height, width)
            weight_ten = torch.zeros(BATCH_SIZE, num_joints, height, width)
         
            regression_target = target
            torch.manual_seed(123)
            
            for j in range(batch_size):
                for i in range(16):
                    joint_center = regression_target[j, i, :]
                
                    joint_map = heatmap.create_gaussian_map(joint_center, (New_width, New_height), sigma=1)
                    weight=heatmap.create_weights(joint_map)
                    weight_ten[j,i,:,:]=weight
                    joint_mapss[j, i, :, :] = joint_map
            joint_mapss=joint_mapss.transpose(2,3)
            weight_ten=weight_ten.transpose(2,3)
            
            data=data.to(dtype=torch.bfloat16)
            regression_target=regression_target.to(dtype=torch.bfloat16)
            joint_mapss=joint_mapss.to(dtype=torch.bfloat16)
            weight_ten=weight_ten.to(dtype=torch.bfloat16)
            data, regression_target, joint_mapss, weight_ten =  data.to(dev), regression_target.to(dev), joint_mapss.to(dev), weight_ten.to(dev)
            output = Model(data)
            
           
            for param in Model.parameters():
                param.grad = None
            loss=criterion(output,joint_mapss,weight_ten,data)
            loss.backward()
            optimizer.step()
            print(f'Train Epoch: {epoch} [{id}/{len(train_loader)} ({100.*id/len(train_loader):.0f}%)]\tLoss: {loss.item()}')
            del loss
            if(id==50):
                id=0

                break
            id=id+1
