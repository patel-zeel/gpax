import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jax.tree_util as tree_util


def assert_same_pytree(tree1, tree2):
    assert tree_util.tree_structure(tree1) == tree_util.tree_structure(tree2)
    assert jnp.all(ravel_pytree(tree1)[0] == ravel_pytree(tree2)[0])


def assert_approx_same_pytree(tree1, tree2):
    assert tree_util.tree_structure(tree1) == tree_util.tree_structure(tree2)
    assert jnp.allclose(ravel_pytree(tree1)[0], ravel_pytree(tree2)[0], atol=1e-2)
