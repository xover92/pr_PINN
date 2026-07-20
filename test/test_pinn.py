import torch
import torch.nn as nn
import pytest
import pr_PINN.pinn as prp


class quadratic_model_1d(nn.Module):
    def forward(self, x, t):
        return x**2+t


class quadratic_model_2d(nn.Module):
    def forward(self, x, y, t):
        return x**2+y**2+t


class quadratic_model_3d(nn.Module):
    def forward(self, x, y, z, t):
        return x**2+y**2+z**2+t


@pytest.fixture
def quad_dummy_model():
    def get_model(dim):
        if dim == 1:
            return quadratic_model_1d()
        if dim == 2:
            return quadratic_model_2d()
        if dim == 3:
            return quadratic_model_3d()
    return get_model


@pytest.mark.parametrize("x_res_1d, t_res_1d, exp_res_1d", [
    (0.0, 0.0, 0.98),
    (1.0, 0.0, 0.98),
    (0.0, 1.0, 0.98),
    (1.0, 1.0, 2.98),
    (0.5, 0.5, 0.7925)
])
def test_pde_residual_1d(quad_dummy_model, x_res_1d, t_res_1d, exp_res_1d):
    """
    Tests wheter the residual is computed correctly, evaluating in the
    boundaries and in a generic point for the function x^2+t.
    """
    x = torch.tensor([[x_res_1d]], requires_grad=True)
    t = torch.tensor([[t_res_1d]], requires_grad=True)
    residual = prp.pde_residual(x, t=t, model=quad_dummy_model(1))
    assert residual.isclose(torch.tensor([[exp_res_1d]]))


@pytest.mark.parametrize("x_res, y_res, t_res, exp_res", [
    (0.0, 0.0, 0.0, 0.96),
    (1.0, 0.0, 0.0, 0.96),
    (0.0, 1.0, 0.0, 0.96),
    (0.0, 0.0, 1.0, 0.96),
    (1.0, 0.0, 1.0, 2.96),
    (0.0, 1.0, 1.0, 2.96),
    (1.0, 1.0, 0.0, 2.96),
    (1.0, 1.0, 1.0, 6.96),
    (0.5, 0.5, 0.5, 0.96),
])
def test_pde_residual_2d(quad_dummy_model, x_res, y_res, t_res, exp_res):
    """
    Tests wheter the residual is computed correctly, evaluating in the
    boundaries and in a generic point for the function x^2+y^2+t.
    """
    x = torch.tensor([[x_res]], requires_grad=True)
    y = torch.tensor([[y_res]], requires_grad=True)
    t = torch.tensor([[t_res]], requires_grad=True)
    residual = prp.pde_residual(x, y, t=t, model=quad_dummy_model(2))
    assert residual.isclose(torch.tensor([[exp_res]]))


@pytest.mark.parametrize("x_res, y_res, z_res, t_res, exp_res", [
    (0.0, 0.0, 0.0, 0.0, 0.94),
    (1.0, 0.0, 0.0, 0.0, 0.94),
    (0.0, 1.0, 0.0, 0.0, 0.94),
    (0.0, 0.0, 1.0, 0.0, 0.94),
    (1.0, 0.0, 1.0, 0.0, 2.94),
    (0.0, 1.0, 1.0, 0.0, 2.94),
    (1.0, 1.0, 0.0, 0.0, 2.94),
    (1.0, 1.0, 1.0, 0.0, 6.94),
    (0.0, 0.0, 0.0, 1.0, 0.94),
    (1.0, 0.0, 0.0, 1.0, 2.94),
    (0.0, 1.0, 0.0, 1.0, 2.94),
    (0.0, 0.0, 1.0, 1.0, 2.94),
    (1.0, 0.0, 1.0, 1.0, 6.94),
    (0.0, 1.0, 1.0, 1.0, 6.94),
    (1.0, 1.0, 0.0, 1.0, 6.94),
    (1.0, 1.0, 1.0, 1.0, 12.94),
    (0.5, 0.5, 0.5, 0.5, 1.2525),
])
def test_pde_residual_3d(quad_dummy_model, x_res, y_res, z_res,
                         t_res, exp_res):
    """
    Tests wheter the residual is computed correctly, evaluating in the
    boundaries and in a generic point for the function x^2+y^2+z^2+t.
    """
    x = torch.tensor([[x_res]], requires_grad=True)
    y = torch.tensor([[y_res]], requires_grad=True)
    z = torch.tensor([[z_res]], requires_grad=True)
    t = torch.tensor([[t_res]], requires_grad=True)
    residual = prp.pde_residual(x, y, z, t=t, model=quad_dummy_model(3))
    assert residual.isclose(torch.tensor([[exp_res]]))


@pytest.mark.parametrize("x_val, t_val, expected", [
    (0.0, 0.0, 0.25),
    (1.0, 0.0, 2.750890767e-4),
    (0.0, 1.0, 0.4858916454),
    (0.5, 0.5, 0.0270849034)
])
def test_exact_solution_1d(x_val, t_val, expected):
    """
    Tests wheter the exact solution is computed correctly, evaluating in the
    boundaries and in a generic point.
    """
    x = torch.tensor([[x_val]], requires_grad=True)
    t = torch.tensor([[t_val]], requires_grad=True)

    result = prp.exact_solution_1D(x, t)
    assert torch.allclose(result, torch.tensor([[expected]]))


class oracle_model(nn.Module):
    def forward(self, x, t):
        return prp.exact_solution_1D(x, t)


@pytest.fixture
def oracle():
    return oracle_model()


@pytest.mark.parametrize("x_val, t_val", [
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.5)
])
def test_loss_1d(x_val, t_val, oracle):
    """
    Tests whete the loss is computed correclty by employing oracle testing.
    More precisely, it calculates the loss for the exact solution and
    checks wheter it is close to 0, with a tolerance of 1e-4.
    """
    x = torch.tensor([[x_val]])
    t = torch.tensor([[t_val]])
    x.requires_grad = True
    t.requires_grad = True
    loss = prp.loss_function(x, mode='exact', t=t, model=oracle)
    print(f"Loss at ({x_val}, {t_val}): {loss.item()}")
    assert torch.allclose(loss, torch.tensor([[0.0]]), atol=1e-4)
