import torch

x = torch.arange(9)
x_3x3 = x.view(3, 3) # acts on contiguous tensors
print(x_3x3)

x_3x3 = x.reshape(3, 3) # doesn't matter (safe bet)
print(x_3x3)

# example for contiguity 

y = x_3x3.t() # transpose rearranges the array in memory (pointer based). View will no longer work
print(y.contiguous().view(9)) 

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim = 0)) # concatenate

# flatten
z = x1.view(-1)
print(z)

x = torch.rand((64, 2, 5))
z = x.view(64, -1)
print(x.shape)

# switching axes
z = x.permute(0, 2, 1)
print(z.shape)

# unsqueezing (inverse of squeeze)
x = torch.arange(10)
print(x.unsqueeze(0).shape)


