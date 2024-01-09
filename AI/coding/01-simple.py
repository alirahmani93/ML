import dataclasses
import os

import numpy as np
import pandas as pd
import torch

file_name = '../../datasets/breast+cancer+wisconsin+diagnostic/wdbc.data'
arr = np.loadtxt(file_name, delimiter=",", dtype=str)
## WITH Numpy
target = arr[:, 1]

target = [1 if i == 'M' else 0 for i in target]
arr[:, 1] = target
arr = arr.astype('float64')
# print(arr)
tensor = torch.from_numpy(arr)
# print(tensor)

## WITH Pandas
# df = pd.read_csv(file_name, delimiter=",", header=None)
#
# df[df.columns[1]] = df[df.columns[1]].map(lambda x: 1 if x == 'M' else 0)
# df_numpy = df.to_numpy(dtype=float)
# df_tensor = torch.tensor(df_numpy)
# print(df_tensor)


print('mean :', torch.mean(tensor, dim=0))
print('std :', torch.std(tensor, dim=0))
print('max :', torch.max(tensor, dim=0))
print('argmax :', torch.argmax(tensor, dim=0))
print('min :', torch.min(tensor, dim=0))
print('argmin :', torch.argmin(tensor, dim=0))

@frozenset
@dataclasses.dataclass
class D:
    a:int