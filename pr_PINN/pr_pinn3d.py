import torch
import torch.nn as nn
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


class PINN_3d(nn.Module):
    """
    The PINN architecture in 3D.

    Parameters
    ----------
    n_neurons:int
    Number of neurons
    hidden (nn.Sequential)
    the neural network architecture itself, which is
    composed by its 4 layers and their Tanh activation functions.
    """

    def __init__(self, n_neurons: int = 50):
        super(PINN_3d, self).__init__()
        self.hidden = nn.Sequential(  # 4 layers neural network
            nn.Linear(4, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                z: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        The core method of the PINN, which approximates u(x, y, z, t).

        Parameters
        ----------
        x:torch.Tensor
            The x spatial coordinates
        y:torch.Tensor
            The y spatial coordinates
        z:torch.Tensor
            The z spatial coordinates
        t:torch.Tensor
            The temporal coordinates

        Returns
        -------
        u: torch.Tensor
            the approximation of u(x, y, z, t)
        """

        inputs = torch.cat([x, y, z, t], dim=1)  # concatenate the inputs
        return self.hidden(inputs)


def pde_residual_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                    t: torch.Tensor, model: nn.Module,
                    D: float = 0.1, R: float = 1) -> torch.Tensor:
    """
    Calculates the residual of the KPP-Fisher equation in 3D.

    Parameters
    ----------
    x:torch.Tensor
        The x spatial coordinates
    y:torch.Tensor
        The y spatial coordinates
    z:torch.Tensor
        The z spatial coordinates
    t:torch.Tensor
        The temporal coordinates
    D:float=0.01
        The diffusion coefficient
    R:float=1
        The reaction rate

    Returns
    -------
    residual:torch.Tensor
        The residual of the KPP-Fisher equation
    """

    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    t.requires_grad = True
    u = model(x, y, z, t)

    u_t = torch.autograd.grad(u, t, torch.ones_like(
        u), create_graph=True)[0]  # du/dt
    u_x = torch.autograd.grad(u, x, torch.ones_like(
        u), create_graph=True)[0]  # du/dx
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(
        u), create_graph=True)[0]  # d^2u/dx^2
    u_y = torch.autograd.grad(u, y, torch.ones_like(
        u), create_graph=True)[0]  # du/dy
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(
        u), create_graph=True)[0]  # d^2u/dy^2
    u_z = torch.autograd.grad(u, z, torch.ones_like(
        u), create_graph=True)[0]  # du/dz
    u_zz = torch.autograd.grad(u_z, z, torch.ones_like(
        u), create_graph=True)[0]  # d^2u/dz^2

    residual = u_t-D*u_xx-D*u_yy-D*u_zz-R*u*(1-u)
    return residual


def loss_function_3d(x: torch.Tensor,
                     y: torch.Tensor,
                     z: torch.Tensor,
                     t: torch.Tensor,
                     model: nn.Module) -> float:
    """
    Computes the loss function as the sum of the initial conditions loss,
    boundary conditions loss and the residual loss.

    Parameters
    ----------
    x:torch.Tensor
        The x spatial coordinates
    y:torch.Tensor
        The y spatial coordinates
    z:torch.Tensor
        The z spatial coordinates
    t:torch.Tensor
        The temporal coordinates
    model:nn.Module
        The approximated u(x, y, z, t)

    Returns
    -------
    loss:float
        The loss function
    """

    # adapt x,y,t to calculate the residual
    x_train, y_train, z_train, t_train = torch.meshgrid(
        x.squeeze(), y.squeeze(), z.squeeze(), t.squeeze(), indexing='ij')
    x_train = x_train.reshape(-1, 1)
    # x_train.requires_grad = True
    y_train = y_train.reshape(-1, 1)
    z_train = z_train.reshape(-1, 1)

    t_train = t_train.reshape(-1, 1)
    # t_train.requires_grad = True

    # initial condition loss
    u_pr = model(x, y, z, torch.zeros_like(x))
    u_ex = torch.zeros_like(x)
    loss_ic = torch.mean((u_pr-u_ex)**2)

    # boundary condition loss
    u_0_pr = model(torch.zeros_like(t), torch.zeros_like(t),
                   torch.zeros_like(t), t)
    u_0_ex = torch.zeros_like(t)
    u_1_pr = model(torch.full_like(t, 1), torch.full_like(
        t, 1), torch.full_like(t, 1), t)
    u_1_ex = torch.full_like(t, 1)
    loss_bc = torch.mean((u_0_pr-u_0_ex)**2)+torch.mean((u_1_pr-u_1_ex)**2)

    # residual loss
    loss_pde = torch.mean(pde_residual_3d(
        x_train, y_train, z_train, t_train, model)**2)

    # total loss
    return loss_bc+loss_ic+loss_pde


def training_loop_3D(n_epochs: int, n_neurons: int) -> nn.Module:
    """
    Runs n_epochs loops to train the model as defined in pr_PINN_3d

    Parameters
    ----------
    n_epochs:int
    The number of epochs for the training loop.
    n_neurons:int
    The numer of neurons per layer.

    Returns
    -------
    model:nn.Module
    The trained PINN.
    """

    model = PINN_3d(n_neurons)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.apply(lambda m: m.reset_parameters()
                # reinitialize the model
                if hasattr(m, 'reset_parameters') else None)

    # generating training points
    x = torch.linspace(0, 1, 20).view(-1, 1)
    y = torch.linspace(0, 1, 20).view(-1, 1)
    z = torch.linspace(0, 1, 20).view(-1, 1)
    t = torch.linspace(0, 1, 20).view(-1, 1)

    for epoch in range(n_epochs):
        model.train()
        loss = loss_function_3d(x, y, z, t, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

    #    if (epoch) % 25 == 0 and (epoch) != 0:
    #        text += f'epoch={epoch}, loss={loss.item():.4f}\n'

    # if (epoch) == 0:
    #    print("Starting.")
    #    if (epoch) == n_epochs-1:
    #        text += f'the final loss is: {loss}'
    # return text


def generate_plot_3d(n_epocs: int, n_neurons: int) -> Figure:
    """
    Runs the loop and then generates a voxel
    plot of solution obtained by the PINN on a fixed time t=0.5.

    Parameters
    ----------
    n_epochs:int
    The number of epochs for the loop
    n_neurons:int
    The numer of neurons per layer.

    Returns
    -------
    fig:Figure
    Returns the figure for gradio to show.
    """

    model = training_loop_3D(n_epocs, n_neurons)

    x_test = torch.linspace(0, 1, 100).view(-1, 1)
    y_test = torch.linspace(0, 1, 100).view(-1, 1)
    z_test = torch.linspace(0, 1, 100).view(-1, 1)
    t_test = torch.linspace(0, 1, 100).view(-1, 1)
    x_test, y_test, z_test = torch.meshgrid(
        x_test.squeeze(), y_test.squeeze(), z_test.squeeze(), indexing='ij')
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    z_test = z_test.reshape(-1, 1)
    t_test = t_test.reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_test, y_test, z_test,
                       torch.full_like(x_test, 0.5)).numpy()

    x_test = x_test.numpy().reshape(100, 100, 100)
    y_test = y_test.numpy().reshape(100, 100, 100)
    z_test = z_test.numpy().reshape(100, 100, 100)
    u_pred = u_pred.reshape(100, 100, 100)

    ds = 5
    x_ds = x_test[::ds, ::ds, ::ds]
    y_ds = y_test[::ds, ::ds, ::ds]
    z_ds = z_test[::ds, ::ds, ::ds]
    u_ds = u_pred[::ds, ::ds, ::ds]

    grid_shape = x_ds.shape
    voxels = np.ones(grid_shape, dtype=bool)

    hole = (x_ds > 0.6) & (y_ds < 0.4) & (z_ds > 0.6)
    voxels[hole] = False

    norm = plt.Normalize(u_ds.min(), u_ds.max())
    colors = cm.viridis(norm(u_ds))
    colors[..., 3] = 1.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0.2)

    return fig
