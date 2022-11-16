import jax.tree_util as jtu

convert = jtu.register_pytree_node_class


@convert
class ABC:
    def __init__(self):
        self.a = [1.0, 2.0]

    def tree_flatten(self):
        return (self.a,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class PQR(ABC):
    pass


a = ABC()
b = PQR()

print()
