class Base:
    """
    This class provides a skeleton for all GPAX classes.
    """

    def __call__(self, params):
        return self.call(params)

    def call(self, params):
        raise NotImplementedError("Must be implemented by subclass")

    def __initialise_params__(self, key, X):
        raise NotImplementedError("Must be implemented by subclass")

    def initialise_params(self, key, X):
        return self.__initialise_params__(key, X)

    def __get_bijectors__(self):
        raise NotImplementedError("Must be implemented by subclass")

    def get_bijectors(self):
        return self.__get_bijectors__()
