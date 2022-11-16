import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp
import gpax.bijectors as gb

import pytest

key = jax.random.PRNGKey(0)
samples = [sample for sample in 10 * jax.random.normal(key, (20,))]


@pytest.mark.parametrize("sample", samples)
@pytest.mark.parametrize("bijector", [gb.Exp(), gb.Log(), gb.Identity(), gb.Softplus(), gb.SquarePlus()])
def test_bijector(sample, bijector):
    tol = 1e-2
    if bijector.in_lower <= sample <= bijector.in_upper:
        transformed_sample = bijector.forward(sample)
        assert bijector.out_lower < transformed_sample < bijector.out_upper
        re_inverse = bijector.inverse(transformed_sample)
        if isinstance(bijector, (gb.Softplus, gb.SquarePlus)):
            if re_inverse == -jnp.inf:
                if sample < -18:
                    assert True
                else:
                    assert False
            else:
                assert re_inverse == pytest.approx(sample, abs=tol)
        else:
            assert re_inverse == pytest.approx(sample, abs=tol)


def test_white_bijector():
    pass
