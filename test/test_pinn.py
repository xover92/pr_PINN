import torch
import torch.nn as nn
import math
import pytest
import pr_PINN.pinn as prp
import itertools

# lhs_sample_generator: shape checking
# lhs for sphere: shape and value check
# pde_residual: tested
# neumann_condition: tested
# neumann for sphere: tested
# dirichlet_condition: tested
# loss_function: tested in 1d by itself and in other dimensions
#                by dependencies
# loss for sphere: tested dependencies
# exact_solution_1d: tested
# training_loop: tested by dependencies
# solve with fipy: not tested (depends on native funcs)
# generate_plot: tested dependencies, done branching test
#  (also done sphere part)


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


x_vals = [0.0, 1.0, 0.5]
y_vals = [0.0, 1.0, 0.5]
z_vals = [0.0, 1.0, 0.5]
t_vals = [0.0, 1.0, 0.5]

test_cases_pde1d = [
    (x, t, 1-0.02-(x**2+t)*(1-x**2-t))
    for x, t in itertools.product(x_vals, t_vals)
]


@pytest.mark.parametrize("x_res_1d, t_res_1d, exp_res_1d", test_cases_pde1d)
def test_pde_residual_1d(quad_dummy_model, x_res_1d, t_res_1d, exp_res_1d):
    """
    Tests wheter the residual is computed correctly, evaluating in the
    boundaries and in a generic point for the function x^2+t.
    """
    x = torch.tensor([[x_res_1d]], requires_grad=True)
    t = torch.tensor([[t_res_1d]], requires_grad=True)
    residual = prp.pde_residual(x, t=t, model=quad_dummy_model(1))
    assert residual.isclose(torch.tensor([[exp_res_1d]]))  # nosec B101


test_cases_pde2d = [
    (x, y, t, 1-0.04-(x**2+y**2+t)*(1-x**2-y**2-t))
    for x, y, t in itertools.product(x_vals, y_vals, t_vals)
]


@pytest.mark.parametrize("x_res, y_res, t_res, exp_res", test_cases_pde2d)
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


test_cases_pde3d = [
    (x, y, z, t, 1-0.06-(x**2+y**2+z**2+t)*(1-x**2-y**2-z**2-t))
    for x, y, z, t in itertools.product(x_vals, y_vals, z_vals, t_vals)
]


@pytest.mark.parametrize("x_res, y_res, z_res, t_res, exp_res",
                         test_cases_pde3d)
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


test_cases_exact_sol = [(x, t, (1+math.exp((0.06**-0.5)*x-5*t/6))**-2)
                        for x, t in itertools.product(x_vals, t_vals)]


@pytest.mark.parametrize("x_val, t_val, expected", test_cases_exact_sol)
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


test_cases_loss1d = list(itertools.product(x_vals, t_vals))


@pytest.mark.parametrize("x_val, t_val", test_cases_loss1d)
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


@pytest.mark.parametrize("x_val, t_val", test_cases_loss1d)
def test_loss_neumann(x_val, t_val, quad_dummy_model):
    x = torch.Tensor([[x_val]])
    t = torch.Tensor([[t_val]])
    x.requires_grad = True
    t.requires_grad = True
    loss = prp.neumann_condition(x, t=t, model=quad_dummy_model(1))
    print(f'{loss}')
    assert torch.allclose(loss, torch.tensor([[4.0]]), atol=1e-4)  # nosec B101


test_cases_loss2d = list(itertools.product(x_vals, y_vals, t_vals))


@pytest.mark.parametrize("x_res, y_res, t_res", test_cases_loss2d)
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


test_cases_loss3d = list(itertools.product(x_vals, y_vals, z_vals, t_vals))


@pytest.mark.parametrize("x_res, y_res, z_res, t_res", test_cases_loss3d)
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


test_cases_dirichlet1d = [(x, t, t, 1+t)
                          for x, t in itertools.product(x_vals, t_vals)]


@pytest.mark.parametrize("x_val, t_val, exp_val_x0, exp_val_x1",
                         test_cases_dirichlet1d)
def test_loss_dirichlet_1d(x_val, t_val, exp_val_x0, exp_val_x1,
                           quad_dummy_model):
    x = torch.Tensor([[x_val]])
    t = torch.Tensor([[t_val]])
    x.requires_grad = True
    t.requires_grad = True
    loss = prp.dirichlet_condition(
        x, t=t, model=quad_dummy_model(1),
        value_x0=exp_val_x0, value_x1=exp_val_x1)
    assert torch.allclose(loss, torch.tensor([[0.0]]), atol=1e-4)  # nosec B101


test_cases_dirichlet2d = [(x, y, t, y**2+t, y**2+1+t, x**2+t, x**2+1+t)
                          for x, y, t in itertools.product(x_vals,
                                                           y_vals, t_vals)]


@pytest.mark.parametrize(
    "x_val, y_val, t_val, exp_val_x0, exp_val_x1, exp_val_y0, exp_val_y1",
    test_cases_dirichlet2d)
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


test_cases_dirichlet3d = [(x, y, z, t, y**2+z**2+t, y**2+z**2+1+t, x**2+z**2+t,
                           x**2+z**2+1+t, x**2+y**2+t, x**2+y**2+1+t)
                          for x, y, z, t in itertools.product(x_vals,
                                                              y_vals,
                                                              z_vals, t_vals)]


@pytest.mark.parametrize(
    "x_val, y_val, z_val, t_val, x0, x1, y0, y1, z0, z1",
    test_cases_dirichlet3d)
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


def test_lhs_sphere_shape_and_properties():
    for dim in [2, 3]:
        points = prp.lhs_sample_generator_sphere_boundary(100, dim)
        assert len(points) == dim  # nosec B101
        radius = 0
        for point in points:
            assert point.shape == (100, 1)  # nosec B101
            assert point.requires_grad is True  # nosec B101
            assert torch.all(point <= 1) and torch.all(
                point >= -1)  # nosec B101
            radius += point**2
        assert torch.allclose(radius, torch.ones_like(radius))  # nosec B101


def test_lhs__in_sphere_shape_and_properties():
    for dim in [2, 3]:
        points = prp.lhs_sample_generator_sphere_inside(100, dim)
        assert len(points) == dim+1  # nosec B101
        radius = 0
        for point in points:
            assert point.shape == (100, 1)  # nosec B101
            assert point.requires_grad is True  # nosec B101
            assert torch.all(point <= 1) and torch.all(
                point >= -1)  # nosec B101
        for point in points[:-1]:
            radius += point**2

        assert torch.all(points[-1] >= 0.0)  # nosec B101
        assert torch.all(radius <= 1.0)  # nosec B101
        assert torch.all(radius >= 0.0)  # nosec B101


test_cases_sphere2d = [(x, y, t, (2*x**2+2*y**2)**2)
                       for x, y, t in itertools.product(x_vals,
                                                        y_vals, t_vals)]


@pytest.mark.parametrize("x_val, y_val, t_val, exp_val", test_cases_sphere2d)
def test_neumann_sphere_2d(x_val, y_val, t_val, exp_val, quad_dummy_model):
    x = torch.Tensor([[x_val]])
    y = torch.Tensor([[y_val]])
    t = torch.Tensor([[t_val]])
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    loss = prp.neumann_condition_sphere(
        x, y, t=t, model=quad_dummy_model(2))
    assert torch.allclose(loss, torch.tensor(
        [[exp_val]]), atol=1e-4)  # nosec B101


test_cases_sphere3d = [(x, y, z, t, (2*x**2+2*y**2+2*z**2)**2)
                       for x, y, z, t in itertools.product(x_vals,
                                                           y_vals, z_vals,
                                                           t_vals)]


@pytest.mark.parametrize("x_val, y_val, z_val, t_val, exp_val",
                         test_cases_sphere3d)
def test_neumann_sphere_3d(x_val, y_val, z_val, t_val, exp_val,
                           quad_dummy_model):
    x = torch.Tensor([[x_val]])
    y = torch.Tensor([[y_val]])
    z = torch.Tensor([[z_val]])
    t = torch.Tensor([[t_val]])
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    t.requires_grad = True
    loss = prp.neumann_condition_sphere(
        x, y, z, t=t, model=quad_dummy_model(3))
    assert torch.allclose(loss, torch.tensor(
        [[exp_val]]), atol=1e-4)  # nosec B101


def test_generate_plot_dirichlett_neumann():
    for dim in [1, 2, 3]:
        for mode in ['dirichlet', 'neumann']:
            fig, l2_loss_text = prp.generate_plot(
                2, 2, 25, dim, mode, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
            loss_value = float(l2_loss_text.split('=')[1])
            assert not math.isnan(loss_value)  # nosec B101
            assert loss_value >= 0  # nosec B101


def test_generate_plot_exact():
    fig, l2_loss_text = prp.generate_plot(
        2, 2, 25, 1, 'exact', 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    loss_value = float(l2_loss_text.split('=')[1])
    assert not math.isnan(loss_value)  # nosec B101
    assert loss_value >= 0  # nosec B101


def test_generate_plot_sphere():
    for dim in [2, 3]:
        fig, l2_loss_text = prp.generate_plot(
            2, 2, 25, dim, 'sphere', 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
