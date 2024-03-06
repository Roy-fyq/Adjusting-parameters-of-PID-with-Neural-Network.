import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from PID_INC import PID_INC


class pid_loss(nn.Module):
    def __init__(self, min, max, dt) -> None:
        super(pid_loss, self).__init__()
        self.min = min
        self.max = max
        self.dt = dt 
        self.ei = 0
        self.ed = 0
        self.pre_err = 0

    def forward(self, input, measure, target, abnormal):
        kp, ki, kd = input
        error = target - measure
        u = kp*error + ki*self.ei + kd*self.ed
        if self.min<u<self.max:
            self.ei = self.ei + error*self.dt
            self.ed = (error - self.pre_err)/self.dt
            self.pre_err = error
        else:
            u = abnormal
        return u

class BP_PID(nn.Module):
    def __init__(self, inputchannels, hiddenchannel, outputchannels) -> None:
        super(BP_PID, self).__init__()

        self.inputlayer = nn.Sequential(nn.Linear(inputchannels, hiddenchannel),
                                        nn.ReLU())
        self.hiddenlayers = nn.Sequential(nn.Linear(hiddenchannel, hiddenchannel*2),
                                         nn.ReLU(),
                                         nn.Linear(hiddenchannel*2, hiddenchannel),
                                         nn.ReLU())
        self.outputlayer = nn.Sequential(nn.Linear(hiddenchannel, outputchannels),
                                         nn.ReLU())
                                         
    def forward(self, x):
        x = self.inputlayer(x)
        x = self.hiddenlayers(x)
        x = self.outputlayer(x)
        return x
    
if __name__ == "__main__":
    model = BP_PID(3, 30, 3)
    opt = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=0.00)
    loss_func = pid_loss(-100, 100, 0.1)
    y = 5    # 定义初始值
    target = 20    # 定义目标值
    data_in = torch.tensor([target, 5, target-y], dtype=torch.float32) #输入为目标值，实际值，误差
    y_list = [5]
    T_list = [0]
    k_list = []
    loss1 = target - y
    epoch = 0
    # for epoch in range(100):
    while loss1 > 1e-5: 
        opt.zero_grad()
        out = model(data_in)
        loss = loss_func(out, torch.tensor(y, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))
        loss.requires_grad_(True)
        loss.backward()
        opt.step()
        data_in = torch.tensor([target, 5+loss.item(), target-(5+loss.item())], dtype=torch.float32)
        y = y+loss.item()   # 这里更改为风机的转速
        print(loss.item())
        T_list.append((epoch+1)*0.1)
        y_list.append(y)
        k_list.append(out.detach().numpy())
        loss1 = np.abs(target-y)
        epoch += 1
        if loss.item()<1e-7:
            break
    a = np.array(k_list)
    y_bp_df = pd.DataFrame({'T': T_list, 'BP_PID_OUT': y_list})
    k_bp_df = pd.DataFrame({'T': T_list[1:], 'KP': a[:,0], 'KI': a[:,1], 'KD': a[:,2]})
    y_bp_df.to_csv('/home/roy/Code/2023Code/pid-LSTM/result/BP_PID_OUT.csv', index=None)
    k_bp_df.to_csv('/home/roy/Code/2023Code/pid-LSTM/result/K_BP.csv', index=None)
    plt.plot(T_list[1:], a[:,], label=['kp', 'ki', 'kd'])
    plt.legend()
    plt.savefig('/home/roy/Code/2023Code/pid-LSTM/result/K.eps', dpi=600)
    plt.show()