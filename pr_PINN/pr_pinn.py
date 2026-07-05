import torch
import torch.nn as nn


class PINN_1d(nn.Module):
    """
    The PINN architecture in 1D.

    Parameters
    ----------
    n_neurons:int
    Number of neurons
    hidden (nn.Sequential)
    the neural network architecture itself, which is
    composed by its 4 layers and their Tanh activation functions.
    """

    def __init__(self, n_neurons: int = 50):
        super(PINN_1d, self).__init__()
        self.hidden = nn.Sequential(  # 4 layers neural network
            nn.Linear(2, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        The core method of the PINN, which approximates u(x,t).

        Parameters
        ----------
        x:torch.Tensor
            The spatial coordinates
        t:torch.Tensor
            The temporal coordinates

        Returns
        -------
        u: torch.Tensor
            the approximation of u(x,t)
        """

        inputs = torch.cat([x, t], dim=1)  # concatenate the inputs
        return self.hidden(inputs)


def pde_residual_1d(x: torch.Tensor,
                    t: torch.Tensor, model: nn.Module,
                    D: float = 0.1, R: float = 1) -> torch.Tensor:
    """
    Calculates the residual of the KPP-Fisher equation in 1D.

    Parameters
    ----------
    x:torch.Tensor
        The spatial coordinates
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
    t.requires_grad = True
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, torch.ones_like(
        u), create_graph=True)[0]  # du/dt
    u_x = torch.autograd.grad(u, x, torch.ones_like(
        u), create_graph=True)[0]  # du/dx
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(
        u), create_graph=True)[0]  # d^2u/dx^2

    residual = u_t-D*u_xx-R*u*(1-u)
    return residual


def exact_solution_1D(x: torch.Tensor,
                      t: torch.Tensor,
                      D: float = 0.1,
                      R: float = 1) -> torch.Tensor:
    """
    Calculates the exact solution of the KPP-Fisher equation in 1D.

    Parameters
    ----------
    x:torch.Tensor
        The spatial coordinates
    t:torch.Tensor
        The temporal coordinates
    D:float=0.01
        The diffusion coefficient
    R:float=1
        The reaction rate

    Returns
    -------
    exact_solution:torch.Tensor
        The exact solution of the KPP-Fisher equation in 1D
    """

    exponent = (((R/(2*D))**0.5)*(x-t*(2*R*D)**0.5))
    exp = torch.exp(exponent)
    exact_solution = 1/(1+exp)
    return exact_solution


def loss_function_1d(x: torch.Tensor,
                     t: torch.Tensor,
                     model: nn.Module) -> float:
    """
    Computes the loss function as the sum of the initial conditions loss,
    boundary conditions loss and the residual loss.

    Parameters
    ----------
    x:torch.Tensor
        The spatial coordinates
    t:torch.Tensor
        The temporal coordinates
    model:nn.Module
        The approximated u(x,t)

    Returns:
    --------
    loss:float
        The loss function
    """

    # adapt x,t to calculate the residual
    x_train, t_train = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='xy')
    x_train = x_train.reshape(-1, 1)
    # x_train.requires_grad = True
    t_train = t_train.reshape(-1, 1)
    # t_train.requires_grad = True

    # initial condition loss
    u_pr = model(x, torch.zeros_like(x))
    u_ex = exact_solution_1D(x, torch.zeros_like(x))
    loss_ic = torch.mean((u_pr-u_ex)**2)

    # boundary condition loss
    u_0_pr = model(torch.zeros_like(t), t)
    u_0_ex = exact_solution_1D(torch.zeros_like(t), t)
    u_1_pr = model(torch.full_like(t, 1), t)
    u_1_ex = exact_solution_1D(torch.full_like(t, 1), t)
    loss_bc = torch.mean((u_0_pr-u_0_ex)**2)+torch.mean((u_1_pr-u_1_ex)**2)

    # residual loss
    loss_pde = torch.mean(pde_residual_1d(x_train, t_train, model)**2)

    # total loss
    return loss_bc+loss_ic+loss_pde
