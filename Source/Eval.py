def evaluate(model, dataloader):
    #model.eval() # Turned off due to BatchNorm layer anomaly in pytorch.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats=False
    total_loss = 0
    total_samples = 0
    x=0
    num_joints = len(joint_cols)
    num_body_parts = len(body_parts)
    height, width = New_width, New_height
    num_joints=16
    batch_size=BATCH_SIZE
    with torch.no_grad():
        for data, target in dataloader:
            data=data.unsqueeze(1)
            joint_mapss = torch.zeros(BATCH_SIZE, 16, New_width, New_height)
            
            target=target.reshape(BATCH_SIZE,16,2)
           
            
            joint_mapss = torch.zeros(BATCH_SIZE, num_joints, height, width)
            weight_ten = torch.zeros(BATCH_SIZE, num_joints, height, width)
            
            torch.manual_seed(123)
            
            for j in range(batch_size):
                for i in range(16):
                    joint_center = target[j, i, :]
                    joint_map = create_gaussian_map(joint_center, (New_width, New_height), sigma=1)
                    weight=create_weights(joint_map)
                    weight_ten[j,i,:,:]=weight
                    joint_mapss[j, i, :, :] = joint_map
            joint_mapss=joint_mapss.transpose(2,3)
            weight_ten=weight_ten.transpose(2,3)
            
            # Forward pass
            data, target, joint_mapss, weight_ten = data.to(dev), target.to(dev), joint_mapss.to(dev), weight_ten.to(dev)
            output = model(data)
            loss=criterion(output,joint_mapss,weight_ten,data)

            # Update statistics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            #print(x)
            
    # Compute average loss over all samples
    avg_loss = total_loss / total_samples
    print('Eval loss: {:.4f}'.format(avg_loss))
    return avg_loss