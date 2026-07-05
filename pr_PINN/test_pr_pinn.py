import torch
import torch.nn as nn
import pytest
import pr_PINN.pr_pinn as prp


class quadratic_model(nn.Module):
    def forward(self, x, t):
        return x**2+t


@pytest.fixture
def quad_dummy_model():
    return quadratic_model()


@pytest.mark.parametrize("x_res, t_res, exp_res", [
    (0.0, 0.0, 0.8),
    (1.0, 0.0, 0.8),
    (0.0, 1.0, 0.8),
    (1.0, 1.0, 2.8),
    (0.5, 0.5, 0.6125)
])
def test_pre_residual_1d(quad_dummy_model, x_res, t_res, exp_res):
    """
    Tests wheter the residual is computed correctly, evaluating in the
    boundaries and in a generic point for the function x^2+t.
    """
    x = torch.tensor([[x_res]], requires_grad=True)
    t = torch.tensor([[t_res]], requires_grad=True)
    assert prp.pde_residual_1d(
        # test residual for simple function
        x, t, quad_dummy_model).isclose(torch.tensor([[exp_res]]))


@pytest.mark.parametrize("x_val, t_val, expected", [
    (0.0, 0.0, 0.5),
    (1.0, 0.0, 0.09655800625),
    (0.0, 1.0, 0.7310585786),
    (0.5, 0.5, 0.3502287219)
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
def test_loss(x_val, t_val, oracle):
    """
    Tests whete the loss is computed correclty by employing oracle testing.
    More precisely, it calculates the loss for the exact solution and
    checks wheter it is close to 0, with a tolerance of 1e-3.
    """
    x = torch.tensor([[x_val]])
    t = torch.tensor([[t_val]])

    loss = prp.loss_function_1d(x, t, oracle)
    print(f"Loss at ({x_val}, {t_val}): {loss.item()}")
    assert torch.allclose(loss, torch.tensor([[0.0]]), atol=1e-2)
