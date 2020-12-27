class BaseModel(object):

    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(self):
        self.name = f'{self.__class__.__name__}'
        # self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')

    def loss(self):  # pylint: disable=no-self-use
        pass

    def optimizer(self, lr, **kwargs):  # pylint: disable=no-self-use
        pass

    def scheduler(self):
        pass

    def training_step(self, batch):
        pass

    def evaluate(self, batch, device):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        pass

    def metrics(self):  # pylint: disable=no-self-use
        pass

    def calculate_metrics(self, batch, device):
        pass

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
