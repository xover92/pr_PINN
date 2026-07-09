import torch
import torch.nn as nn
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pr_PINN.pr_pinn import lhs_sample_generator
from fipy import CellVariable, Grid3D, TransientTerm, DiffusionTerm


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

    # initial condition loss
    u_pr = model(x, y, z, torch.zeros_like(x))
    u_ex = torch.zeros_like(x)
    loss_ic = torch.mean((u_pr-u_ex)**2)

    # boundary condition loss
    u_0_pr_0yz = model(torch.zeros_like(t), y,
                       z, t)
    u_0_ex_0yz = torch.zeros_like(t)
    u_0_pr_x0z = model(x, torch.zeros_like(t),
                       z, t)
    u_0_ex_x0z = torch.zeros_like(t)
    u_0_pr_xy0 = model(x, y, torch.zeros_like(t),
                       t)
    u_0_ex_xy0 = torch.zeros_like(t)
    u_1_pr_1yz = model(torch.full_like(t, 1), y, z, t)
    u_1_ex_1yz = torch.full_like(t, 1)
    u_1_pr_x1z = model(x, torch.full_like(t, 1), z, t)
    u_1_ex_x1z = torch.full_like(t, 1)
    u_1_pr_xy1 = model(x, y, torch.full_like(t, 1), t)
    u_1_ex_xy1 = torch.full_like(t, 1)
    loss_bc = torch.mean((u_0_pr_0yz-u_0_ex_0yz)**2) + \
        torch.mean((u_0_pr_xy0-u_0_ex_xy0)**2) + \
        torch.mean((u_0_pr_x0z-u_0_ex_x0z)**2) + \
        torch.mean((u_1_pr_1yz-u_1_ex_1yz)**2) + \
        torch.mean((u_1_pr_x1z-u_1_ex_x1z)**2) + \
        torch.mean((u_1_pr_xy1-u_1_ex_xy1)**2)

    # residual loss
    loss_pde = torch.mean(pde_residual_3d(
        x, y, z, t, model)**2)

    # total loss
    return loss_bc+loss_ic+loss_pde


def solve_with_fipy_3d():

    nx = 20
    ny = nx
    dx = 0.05
    dy = dx
    nz = nx
    dz = dx

    mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)

    phi = CellVariable(name='solution variable', mesh=mesh, value=0.0)

    eq = TransientTerm() == DiffusionTerm(coeff=0.1) + phi*(1-phi)

    phi.constrain(0.0, where=mesh.facesLeft |
                  mesh.facesBottom | mesh.facesFront)
    phi.constrain(1.0, where=mesh.facesRight | mesh.facesTop | mesh.facesBack)

    # steps = 20
    history = []
    # t = 0
    # for step in range(steps):
    eq.solve(var=phi,
             dt=0.5)
    history.append((np.array(phi.value).reshape((nx, ny, nz))))

    return history


def training_loop_3D(n_epochs: int, n_neurons: int,
                     n_points: int) -> nn.Module:
    """
    Runs n_epochs loops to train the model as defined in pr_PINN_3d

    Parameters
    ----------
    n_epochs:int
    The number of epochs for the training loop.
    n_neurons:int
    The numer of neurons per layer.
    n_points:int
    The number of points for LHS sampling.

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
    x, y, z, t = lhs_sample_generator(n_points, 4)

    loss_list = []

    for epoch in range(n_epochs):
        model.train()
        loss = loss_function_3d(x, y, z, t, model)
        if epoch % 10 == 0:
            loss_list.append([loss.item(), epoch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, loss_list


def generate_plot_3d(n_epocs: int, n_neurons: int, n_points: int) -> Figure:
    """
    Runs the loop and then generates a voxel
    plot of solution obtained by the PINN on a fixed time t=0.5, compared
    with the one obtained by fipy, and the evolution of the loss
    with the epochs.

    Parameters
    ----------
    n_epochs:int
    The number of epochs for the loop
    n_neurons:int
    The numer of neurons per layer.
    n_points:int
    The number of points for LHS sampling.

    Returns
    -------
    fig:Figure
    Returns the figure for gradio to show.
    """

    model, loss_list = training_loop_3D(n_epocs, n_neurons, n_points)
    history = solve_with_fipy_3d()

    x_test = torch.linspace(0, 1, 20).view(-1, 1)
    y_test = torch.linspace(0, 1, 20).view(-1, 1)
    z_test = torch.linspace(0, 1, 20).view(-1, 1)
    t_test = torch.linspace(0, 1, 20).view(-1, 1)
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

    x_test = x_test.numpy().reshape(20, 20, 20)
    y_test = y_test.numpy().reshape(20, 20, 20)
    z_test = z_test.numpy().reshape(20, 20, 20)
    u_pred = u_pred.reshape(20, 20, 20)

    ds = 1
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

    u_exact = history[0]
    u_exact_ds = u_exact[::ds, ::ds, ::ds]

    colors_exact = cm.viridis(norm(u_exact_ds))
    colors_exact[..., 3] = 1.0

    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0.2)

    ax_exact = fig.add_subplot(1, 3, 2, projection='3d')
    ax_exact.voxels(voxels, facecolors=colors_exact,
                    edgecolor='k', linewidth=0.2)
    ax_exact.set_title("Exact (FiPy)")

    losses = [item[0] for item in loss_list]
    epochs = [item[1] for item in loss_list]

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(epochs, losses, label="loss")
    ax2.set_yscale('log')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.grid(True)
    plt.tight_layout()

    return fig
