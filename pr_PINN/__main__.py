#!/usr/bin/python
# -*- coding: utf-8 -*-

import pr_PINN.pr_pinn as prp
import torch
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__author__ = ['Francesco Colombo']
__email__ = ['francesco.colombo10@studio.unibo.it']

torch.manual_seed(42)

model = None
optimizer = None


def training_loop_1D(n_epochs: int, n_neurons: int) -> None:
    """
    Runs n_epochs loops to train the model as defined in pr_pinn

    Parameters
    ----------
    n_epochs:int
    The number of epochs for the training loop.
    n_neurons:int
    The numer of neurons per layer.
    """
    global model, optimizer

    model = prp.PINN_1d(n_neurons)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.apply(lambda m: m.reset_parameters()
                # reinitialize the model
                if hasattr(m, 'reset_parameters') else None)

    # generating training points
    x = torch.linspace(0, 1, 200).view(-1, 1)
    t = torch.linspace(0, 1, 200).view(-1, 1)

    for epoch in range(n_epochs):
        model.train()
        loss = prp.loss_function_1d(x, t, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #    if (epoch) % 25 == 0 and (epoch) != 0:
    #        text += f'epoch={epoch}, loss={loss.item():.4f}\n'

        # if (epoch) == 0:
        #    print("Starting.")
    #    if (epoch) == n_epochs-1:
    #        text += f'the final loss is: {loss}'
    # return text


def generate_plot_1d(n_epocs: int, n_neurons: int) -> Figure:
    """
    Runs the loop and then generates a side by side contour
    plot of the correct solution and the one obtained by the PINN.

    Parameters
    ----------
    n_epochs:int
    The number of epochs for the loop
    n_neurons:int
    The numer of neurons per layer.

    Returns
    -------
    Figure
    Returns the figure for gradio to show.
    """

    training_loop_1D(n_epocs, n_neurons)

    x_test = torch.linspace(0, 1, 100).view(-1, 1)
    t_test = torch.linspace(0, 1, 100).view(-1, 1)
    x_test, t_test = torch.meshgrid(
        x_test.squeeze(), t_test.squeeze(), indexing='xy')
    x_test = x_test.reshape(-1, 1)
    t_test = t_test.reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_test, t_test).numpy()
        u_exact = prp.exact_solution_1D(x_test, t_test).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))

    x_test = x_test.numpy().reshape(100, 100)
    t_test = t_test.numpy().reshape(100, 100)
    u_pred = u_pred.reshape(100, 100)
    u_exact = u_exact.reshape(100, 100)

    c1 = ax1.contourf(x_test, t_test, u_pred, levels=250, cmap='jet')
    fig.colorbar(c1, ax=ax1)
    c2 = ax2.contourf(x_test, t_test, u_exact, levels=250, cmap='jet')
    fig.colorbar(c2, ax=ax2)
    plt.tight_layout()

    return fig


demo = gr.Interface(
    fn=generate_plot_1d,
    inputs=[gr.Slider(maximum=1000), gr.Slider()],
    outputs=gr.Plot(),
    api_name="predict"
)

demo.launch()
