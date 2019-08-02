import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np

from pyangel import angelps
import grpc
from torch.optim.optimizer import Optimizer, required


os.environ['jvm_port']='9005'
os.environ['plasma_name']='/tmp/plasma'
ps = angelps.AngelPs()
ps.batch_size = 1
#hhh = {}
hhh_key = {}
param_id = 0

class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    #print("grad: "+str(type(p.grad))+' p: '+str(type(p)))
                    #print("data: " + str(type(p.data))+' grad data: '+str(type(p.grad.data)))
                    p.grad.detach_()
                    p.grad.zero_()
                    key = hhh_key[p]
                    data = ps.pull([key])[0]
                    p.data = torch.from_numpy(data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print(p.data)
                key = hhh_key[p]
                print(d_p.numpy().dtype)
                print(p.data.numpy().dtype)
                ps.push([key], [d_p.numpy().astype(np.float64)])

                #p.data = torch.from_numpy(np.ones_like(p.data.numpy()))
                #new_data = p.data.clone().detach()
                #new_data.add_(-group['lr'], d_p)
                #hhh[p]=new_data
                #p.data.add_(-group['lr'], d_p)
                #print(p.data)

            return loss

batch_size = 100
n_iters = 3000
input_dim = 784
output_dim = 10
lr_rate = 0.001

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

epochs = n_iters / (len(train_dataset) / batch_size)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


model = LogisticRegression(input_dim, output_dim)

criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy

optimizer = SGD(model.parameters(), lr=lr_rate)

for param in model.parameters():
    print(param.size())
for name, param in model.named_parameters():
    print(name)
    print(param.size())
for param in model.parameters():
    key = str(param_id)
    hhh_key[param] = str(param_id)
    param_id += 1
    ps.create_variable(key, param.data.numpy().shape, np.float)
ps.init()

print(ps.key_matid.items())

iter_num = 0

for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter_num+=1
        if iter_num%500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))
