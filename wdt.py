import jax
import jax.tree_util as jtu
from chex import dataclass


@jtu.register_pytree_node_class
@dataclass
class GP:
    kernel: float

    def tree_flatten(self):
        return (self.kernel,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


cls = GP(kernel=2.0)

print("cls:", cls)
#     def __init__(self, kernel, likelihood, mean, X_inducing=None):
#         self.kernel = kernel
#         self.likelihood = likelihood
#         self.mean = mean
#         self.modules = {"kernel": self.kernel, "likelihood": self.likelihood, "mean": self.mean}
#         self.common_params = {"X_inducing": X_inducing}

#     def post_processing(self):
#         jtu.tree_map(lambda x: x.post_processing(), self.components)

#     def log_prob(self, params):
#         return params["p"] ** 2

#     def flatten(self):
#         return {
#             **jtu.tree_map(lambda x: x.flatten(), self.components),
#             "X_inducing": self.X_inducing,
#         }

#     def constrain(self):
#         jtu.tree_map(lambda x: x.constrain(), self.components)

#     def unconstrain(self):
#         jtu.tree_map(lambda x: x.unconstrain(), self.components)

#     def tree_flatten(self):
#         values, tree_def =
#         return (values, aux)

#     @classmethod
#     def tree_unflatten(cls, aux, params):

#         return cls(**params)


# a = ABC(p=2.0)
# print(jax.tree_util.tree_leaves(a))
# print(jax.tree_util.tree_structure(a))
# print(jax.grad(a.log_prob)(a))


class ABC:
    def __init__(self):
        self.a = [1.0, 2.0]
        print(self.a)
        delattr(self, "a")


p = ABC()
print(p.a)
