""" file description
"""
import logging
import logging.config
from textwrap import dedent

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import yaml
# from rouge_score import rouge_scorer, scoring
# from sacrebleu import corpus_bleu

# from base_model import BaseModel
# from utils import calculate_rouge, calculate_bleu
from translator.base_model import BaseModel
from translator.utils import calculate_rouge, calculate_bleu

DEFAULT_PRETRAINED_HFACE_MODEL_FILEPATH = "opus-mt-en-ru"
DEFAULT_HF_NAME_MODEL_FILEPATH = "Helsinki-NLP/opus-mt-en-ru"
DEFAULT_PRETRAINED_USER_MODEL_FILEPATH = "./data/stored-opus-mt-en-ru"

APPLICATION_NAME = "translator"
DEFAULT_CONFIG_FILEPATH = "translator.conf.yml"
DEFAULT_LOGGING_CONFIG_FILEPATH = "logging.conf.yml"
MAX_LENGTH = 250

with open(DEFAULT_CONFIG_FILEPATH, "r") as fin:
    cfg = yaml.safe_load(fin)

# logger = logging.getLogger()
logger = logging.getLogger(APPLICATION_NAME)  # for current example

# setup_logging()

def setup_config():
    pass

class OPUSModel(BaseModel):

    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(self, tokenizer_filepath: str):
        # self.name = f'{self.__class__.__name__}_{tokenizer_filepath}'
        # self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_filepath)

    def predict(self, text: str) -> str:
        if len(text) > MAX_LENGTH:
            return dedent(f"""Your text has length {len(text)} > {MAX_LENGTH}.
            Please, reduce text for translator""")
        ids = self.tokenizer(text, return_tensors='pt').input_ids
        output = self.model.generate(ids)
        return self.tokenizer.decode(output[0])

    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None):
        pass

    def evaluate(self, x, y, batch_size=16, verbose=False):  # pylint: disable=unused-argument
        pass


    def loss(self, input_text: str, target_text: str) -> float:  # pylint: disable=no-self-use
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids
        target_ids = self.tokenizer(target_text, return_tensors='pt').input_ids
        loss = self.model(input_ids=input_ids, labels=target_ids).loss
        return loss


    def optimizer(self):  # pylint: disable=no-self-use
        pass


    def metrics(self, input_text: str, target_text: str, metrics: list = None):  # pylint: disable=no-self-use
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

def setup_logging():
    """ Setup logging file
    """
    with open(DEFAULT_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))

# def main():

# if __name__ == "__main__":
    # main()


