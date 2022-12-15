## Design Decisions
* Pure PyTorch like design with the following change
    * Return nested dictionary instead of flat. This is easier to manage using `jax.tree_utils`
* \ell, \sigma and \omega should be modelled without squaring them.