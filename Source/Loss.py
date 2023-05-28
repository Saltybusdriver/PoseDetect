from sklearn.metrics import mean_squared_error
class LossCalculation(nn.Module):
    

    def __init__(self):
        super().__init__()
        self.regression_loss_fn = nn.MSELoss(reduction='mean')
    def forward(self, input, joint_mapss,weight,img,depth):
        regression_output = input
        torch.set_printoptions(threshold=torch.inf)
        batch_size, n_classes, height, width = joint_mapss.shape
  
        loss = 0
        regression_output.squeeze(3)
        for i in range(depth):
            outputs=regression_output[:,i,:,:,:]
            output_i = outputs[:,:, :, :]
            target_i = joint_mapss[:, :, :, :]
            loss = loss+  torch.mean(torch.square(output_i-target_i)*weight)
        loss /= 16
        
        return loss
        
   
       
