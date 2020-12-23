""" file description
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_PRETRAINED_HFACE_MODEL_FILEPATH = "opus-mt-en-ru"
DEFAULT_HF_NAME_MODEL_FILEPATH = "Helsinki-NLP/opus-mt-en-ru"
DEFAULT_PRETRAINED_USER_MODEL_FILEPATH = "./store-pretrained"



class OPUSModel(BaseModel):

    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(self, tokenizer_filepath: str):
        # self.name = f'{self.__class__.__name__}_{tokenizer_filepath}'
        # self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_filepath)

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')
    
    def predict(self, text: str) -> str:
        ids = self.tokenizer(text, return_tensors = 'pt').input_ids
        output = self.model.generate(ids)
        return self.tokenizer.decode(output[0])

    def evaluate(self, text: str) -> str:
        ids = self.tokenizer(text, return_tensors = 'pt').input_ids
        output = self.model.generate(ids)
        return self.tokenizer.decode(output[0])

    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None):
        pass  

    def evaluate(self, x, y, batch_size=16, verbose=False):  # pylint: disable=unused-argument
         pass


    def loss(self, input_text: str, target_text: str) -> float:  # pylint: disable=no-self-use
        input_ids = self.tokenizer(input_text, return_tensors = 'pt').input_ids
        target_ids = self.tokenizer(target_text, return_tensors = 'pt').input_ids
        loss = self.model(input_ids=input_ids, labels=target_ids).loss 
        return loss


    def optimizer(self):  # pylint: disable=no-self-use
        pass


    def metrics(self):  # pylint: disable=no-self-use
        pass

    def save_model(self, filepath: str):
        """ Save the trained model
        """
        self.tokenizer.save_pretrained(filepath)
        self.model.save_pretrained(filepath)

    def load_model(self, filepath: str):
        """ Load the trained model
        """
        self.tokenizer.from_pretrained(filepath)
        self.model.from_pretrained(filepath)
