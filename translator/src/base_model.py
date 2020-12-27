class BaseModel(object):

    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(self):
        self.name = f'{self.__class__.__name__}'
        # self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')


    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None):
        pass

    def evaluate(self, x, y, batch_size=16, verbose=False):  # pylint: disable=unused-argument
         pass


    def loss(self):  # pylint: disable=no-self-use
        pass


    def optimizer(self):  # pylint: disable=no-self-use
        pass


    def metrics(self):  # pylint: disable=no-self-use
        pass


    def load_weights(self):
        self.network.load_weights(self.weights_filename)


    def save_weights(self):
        self.network.save_weights(self.weights_filename)


