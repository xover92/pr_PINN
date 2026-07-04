#!/usr/bin/python
# -*- coding: utf-8 -*-

from pr_PINN.__version__ import __version__
import pr_PINN.pr_pinn as prp
import torch

__author__ = ['Francesco Colombo']
__email__ = ['francesco.colombo10@studio.unibo.it']

__all__ = [
  '__version__',
  'pr_pinn'
]

model=prp.PINN_1d()
optimizer =torch.optim.Adam(model.parameters(), lr=0.001)

#generating training points
x=torch.linspace(0, 1, 200).view(-1, 1)
t=torch.linspace(0, 1, 200).view(-1, 1)

n_epochs=1200
for epoch in range(n_epochs):
    model.train()
    loss=prp.loss_function_1d(x, t, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch) % 200==0 and (epoch) !=0:
        print(f'epoch={epoch}, loss={loss.item():.4f}')

    if (epoch)==0:
      print("Starting.")

print(f'the final loss is: {loss}')


