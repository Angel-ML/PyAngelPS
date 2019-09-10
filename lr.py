import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np

from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
import time


hhh = {}
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
                    p.data = hhh[p]

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

                #p.data = torch.from_numpy(np.ones_like(p.data.numpy()))
                new_data = p.data.clone().detach()
                new_data.add_(-group['lr'], d_p)
                hhh[p]=new_data
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
        #outputs = F.softmax(self.linear(x), dim = 1)
        outputs = self.linear(x)
        return outputs


#model = torch.jit.script(LogisticRegression(input_dim, output_dim))
model = LogisticRegression(input_dim, output_dim)
#torch.jit.save(model,'lr.ndl')
#model = torch.jit.load('lr.ndl')
criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy
#criterion = torch.jit.script(torch.nn.CrossEntropyLoss()) # computes softmax and then the cross entropy

optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

for param in model.parameters():
    print(param.size())
for name, param in model.named_parameters():
    print(name)
    print(param.size())

iter_num = 0
start_time = time.time()
b = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        a = time.time()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        b += time.time() - a
        optimizer.step()

        iter_num+=1
        if iter_num%500==0:
            # calculate Accuracy
            print(time.time() - start_time)
            print('model: ',b)
            start_time = time.time()
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                outputs = model(images)
                tmp, predicted = torch.max(outputs.data, 1)
                print(tmp,predicted)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter_num, loss.item(), accuracy))
