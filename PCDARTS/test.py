import torch
import torch.nn as nn

# m = nn.Sigmoid()
# loss = nn.BCELoss()
# input = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# print(input, input.shape)
# print(target, target.shape)
# print(m(input), m(input).shape)

# output = loss(m(input), target)

# output.backward()

a = torch.tensor([0.0865, 0.2454])
b = torch.tensor([1., 0.])
print(a-b)

'''
tensor([[-1.5536,  1.6110],
        [-1.2483,  1.1975]], device='cuda:0', grad_fn=<AddmmBackward>) torch.Size([2, 2])
tensor([1, 0], device='cuda:0') torch.Size([2])

tensor([ 1.7232, -0.7162,  0.2902], requires_grad=True) torch.Size([3])
tensor([1., 1., 0.]) torch.Size([3])
tensor([0.8485, 0.3282, 0.5720], grad_fn=<SigmoidBackward>) torch.Size([3])

'''