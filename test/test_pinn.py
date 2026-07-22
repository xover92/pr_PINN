import torch
import torch.nn as nn
import math
import pytest
import pr_PINN.pinn as prp

# lhs_sample_generator: shape checking
# pde_residual: tested
# neumann_condition: tested
# dirichlet_condition: tested
# loss_function: tested in 1d by itself and in other dimensions
#                by dependencies
# exact_solution_1d: tested
# training_loop: tested by dependencies
# solve with fipy: not tested (depends on native funcs)
# generate_plot: tested dependencies, done branching test

# TODO change the test points logic by using itertools


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
    assert residual.isclose(torch.tensor([[exp_res_1d]]))  # nosec B101


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
    assert residual.isclose(torch.tensor([[exp_res]]))  # nosec B101


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
    assert residual.isclose(torch.tensor([[exp_res]]))  # nosec B101


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
    assert torch.allclose(result, torch.tensor([[expected]]))  # nosec B101


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
    assert torch.allclose(loss, torch.tensor([[0.0]]), atol=1e-4)  # nosec B101


@pytest.mark.parametrize("x_val, t_val", [
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.5)
])
def test_loss_neumann(x_val, t_val, quad_dummy_model):
    x = torch.Tensor([[x_val]])
    t = torch.Tensor([[t_val]])
    x.requires_grad = True
    t.requires_grad = True
    loss = prp.neumann_condition(x, t=t, model=quad_dummy_model(1))
    print(f'{loss}')
    assert torch.allclose(loss, torch.tensor([[4.0]]), atol=1e-4)  # nosec B101


@pytest.mark.parametrize("x_res, y_res, t_res", [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
    (0.5, 0.5, 0.5),
])
def test_loss_neumann_2d(x_res,  y_res, t_res, quad_dummy_model):
    x = torch.Tensor([[x_res]])
    y = torch.Tensor([[y_res]])
    t = torch.Tensor([[t_res]])
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    loss = prp.neumann_condition(x, y, t=t, model=quad_dummy_model(2))
    print(f'{loss}')
    assert torch.allclose(loss, torch.tensor([[8.0]]), atol=1e-4)  # nosec B101


@pytest.mark.parametrize("x_res, y_res, z_res, t_res", [
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0, 0.0),
    (1.0, 1.0, 0.0, 0.0),
    (1.0, 1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0, 1.0),
    (0.0, 1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
    (0.5, 0.5, 0.5, 0.5),
])
def test_loss_neumann_3d(x_res,  y_res, z_res, t_res, quad_dummy_model):
    x = torch.Tensor([[x_res]])
    y = torch.Tensor([[y_res]])
    z = torch.Tensor([[z_res]])
    t = torch.Tensor([[t_res]])
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    t.requires_grad = True
    loss = prp.neumann_condition(x, y, z, t=t, model=quad_dummy_model(3))
    print(f'{loss}')
    assert torch.allclose(loss, torch.tensor(
        [[12.0]]), atol=1e-4)  # nosec B101


@pytest.mark.parametrize("x_val, t_val, exp_val_x0, exp_val_x1", [
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0, 2.0),
    (0.5, 0.5, 0.50, 1.50)
])
def test_loss_dirichlet(x_val, t_val, exp_val_x0, exp_val_x1,
                        quad_dummy_model):
    x = torch.Tensor([[x_val]])
    t = torch.Tensor([[t_val]])
    x.requires_grad = True
    t.requires_grad = True
    loss = prp.dirichlet_condition(
        x, t=t, model=quad_dummy_model(1),
        value_x0=exp_val_x0, value_x1=exp_val_x1)
    assert torch.allclose(loss, torch.tensor([[0.0]]), atol=1e-4)  # nosec B101


@pytest.mark.parametrize(
    "x_val, y_val, t_val, exp_val_x0, exp_val_x1, exp_val_y0, exp_val_y1", [
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0),
        (0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0),
        (1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0),
        (0.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0),
        (1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0),
        (1.0, 1.0, 1.0, 2.0, 3.0, 2.0, 3.0),
        (0.5, 0.5, 0.5, 0.75, 1.75, 0.75, 1.75),
    ])
def test_loss_dirichlet_2d(x_val, y_val, t_val, exp_val_x0, exp_val_x1,
                           exp_val_y0, exp_val_y1,
                           quad_dummy_model):
    x = torch.Tensor([[x_val]])
    y = torch.Tensor([[y_val]])
    t = torch.Tensor([[t_val]])
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    loss = prp.dirichlet_condition(
        x, y, t=t, model=quad_dummy_model(2),
        value_x0=exp_val_x0, value_x1=exp_val_x1,
        value_y0=exp_val_y0, value_y1=exp_val_y1)
    assert torch.allclose(loss, torch.tensor([[0.0]]), atol=1e-4)  # nosec B101


@pytest.mark.parametrize(
    "x_val, y_val, z_val, t_val, x0, x1, y0, y1, z0, z1", [
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0),
        (0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0),
        (0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0),
        (0.0, 1.0, 1.0, 0.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0),
        (1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0),
        (1.0, 1.0, 1.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0),
        (1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 3.0),
        (0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0),
        (0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0, 2.0),
        (1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 2.0, 3.0),
        (0.0, 1.0, 1.0, 1.0, 3.0, 4.0, 2.0, 3.0, 2.0, 3.0),
        (1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0),
        (1.0, 1.0, 1.0, 1.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0),
        (0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0),
    ])
def test_loss_dirichlet_3d(x_val, y_val, z_val, t_val, x0, x1,
                           y0, y1, z0, z1,
                           quad_dummy_model):
    x = torch.Tensor([[x_val]])
    y = torch.Tensor([[y_val]])
    z = torch.Tensor([[z_val]])
    t = torch.Tensor([[t_val]])
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    t.requires_grad = True
    loss = prp.dirichlet_condition(
        x, y, z, t=t, model=quad_dummy_model(3),
        value_x0=x0, value_x1=x1,
        value_y0=y0, value_y1=y1,
        value_z0=z0, value_z1=z1)
    assert torch.allclose(loss, torch.tensor([[0.0]]), atol=1e-4)  # nosec B101


def test_lhs_sample_shape_and_properties():
    for dim in [1, 2, 3]:
        points = prp.lhs_sample_generator(100, dim)
        assert len(points) == dim  # nosec B101
        for point in points:
            assert point.shape == (100, 1)  # nosec B101
            assert point.requires_grad is True  # nosec B101
            assert torch.all(point <= 1) and torch.all(
                point >= 0)  # nosec B101


def test_generate_plot_dirichlett_neumann():
    for dim in [1, 2, 3]:
        for mode in ['dirichlet', 'neumann']:
            fig, l2_loss_text = prp.generate_plot(
                2, 2, 50, dim, mode, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
            loss_value = float(l2_loss_text.split('=')[1])

            assert not math.isnan(loss_value)  # nosec B101
            assert loss_value >= 0  # nosec B101


def test_generate_plot_exact():
    fig, l2_loss_text = prp.generate_plot(
        5, 5, 100, 1, 'exact', 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    loss_value = float(l2_loss_text.split('=')[1])
    assert not math.isnan(loss_value)  # nosec B101
    assert loss_value >= 0  # nosec B101
