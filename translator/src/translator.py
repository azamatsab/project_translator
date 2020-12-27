""" file description
"""
import os
import logging
import logging.config
from textwrap import dedent

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import yaml

from translator.src.base_model import BaseModel
from translator.src.utils import calculate_rouge, calculate_bleu

DEFAULT_CONFIG_FILEPATH = os.sep.join(
    [
        os.path.dirname(__file__), 
        '..', 
        'params', 
        'config.translator.yml',
    ]
)

with open(DEFAULT_CONFIG_FILEPATH, 'r') as fin:
    cfg = yaml.safe_load(fin)

logger = logging.getLogger(__name__)

class OPUSModel(BaseModel):

    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(self, tokenizer_filepath: str):
        self.name = f'{self.__class__.__name__}_{tokenizer_filepath}'
        # self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'
        logger.info("Initialize %s class with pretrained model %s", self.__class__.__name__, tokenizer_filepath)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_filepath)


    def predict(self, text: str) -> str:
        """ Predict text from English to Russian
        """
        if len(text) > cfg['MAX_LENGTH']:
            logger.debug("Try to translate too long sentence")
            return dedent(f"""Your text has length {len(text)} > {cfg['MAX_LENGTH']}.
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
        logger.debug('for request "%s" get loss %g', input_text, loss)
        return loss


    def optimizer(self):  # pylint: disable=no-self-use
        pass


    def metrics(self, input_text: str, target_text: str, metrics: list = None):  # pylint: disable=no-self-use
        pass

    def save_model(self, filepath: str):
        """ Save the trained model
        """
        logger.info('Store the model to path: "%s"', filepath)
        self.tokenizer.save_pretrained(filepath)
        self.model.save_pretrained(filepath)


    def load_model(self, filepath: str):
        """ Load the trained model
        """
        logger.info('Load the model for path: "%s"', filepath)
        self.tokenizer.from_pretrained(filepath)
        self.model.from_pretrained(filepath)

# def main():

# if __name__ == "__main__":
    # main()


