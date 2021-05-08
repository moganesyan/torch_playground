import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# basic initialization
my_tensor = torch.tensor([[1, 2, 3],[4, 5, 6]],
    dtype = torch.float32,
    device = device,
    requires_grad = True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)

# other common initialization methods
x = torch.empty(size = (3, 3))
print(x)

y = torch.zeros((3, 3))
print(y)

z = torch.rand((3, 3))
print(z)

q = torch.arange(start = 0, end = 5, step = 1)
print(q)

u = torch.linspace( start = 0.1, end = 1, steps = 10)
print(u)

m = torch.empty(size = (1, 5)).normal_(mean = 0, std = 1)
print(m)

# how to initialize and convert tensors to types
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short()) # int16
print(tensor.long()) # int64
print(tensor.half()) # float16
print(tensor.float()) # float32
print(tensor.double()) # float64

import numpy as np
np_array = np.zeros((5,5))
tensor_np = torch.from_numpy(np_array)
np_array_back = tensor_np.numpy()
print(np_array_back)

# tensor math
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# addition
z1 = torch.zeros(3, dtype = torch.int64)
torch.add(x,y, out = z1)
print(z1)

z2 = torch.add(x,y)
print(z2)

z3 = x + y
print(z3)

# division 
z = torch.true_divide(x, y) # elementwise
print(z)

# inplace operations
t = torch.zeros(3)
t.add_(x) # underscores imply that the operation is done in place
t+=x

# exponentiation
z = x.pow(2) # elementwise
print(z)

z = x ** 2
print(z)

# simple comparison
z = x > 0 # elementwise
print(z)

# matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
print(x3)

x3 = x1.mm(x2)
print(x3)

# matrix exponentiation
matrix_exp = torch.rand((5, 5))
print(matrix_exp.matrix_power(3))

# elementwise matrix multplication
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(out_bmm)

# examples of broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2 # x2 is expanded along axis=1 (row replication)

# other tensor operations
sum_x = torch.sum(x, dim = 0)
values, idx = torch.max(x, dim = 0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim = 0)
mean_x = torch.mean(x.float(), dim = 0) # computing the mean requires a float tensor

z = torch.eq(x, y) # elementwise compare
torch.sort(y, dim = 0, descending = False) # sorting

z = torch.clamp(x, min = 0, max = 10)

x = torch.tensor([1, 0, 1, 1, 1], dtype = torch.bool)
z = torch.any(x)
print(z)
