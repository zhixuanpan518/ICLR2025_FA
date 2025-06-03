import torch
matrix = torch.zeros((10, 2))
for i in range(2):
    matrix[5*i:5*i+5, i] = 1
print(matrix)