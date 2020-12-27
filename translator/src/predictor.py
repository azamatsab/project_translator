import logging
from textwrap import dedent

from translator.src.base_predictor import Predictor
from translator.src.base_model import BaseModel
# from translator import OPUSModel

MAX_LENGTH = 250

logger = logging.getLogger(__name__)

class CommonPredictor(Predictor):
    """ Instance of OPUSModel transformer.

        Load from
    """
    def __init__(self, model: BaseModel, path_to_model: str = None):
        logger.info(f"Init CommonPredictor with args:  '{model.__name__}' and '{path_to_model}'")
        self.model = model(tokenizer_filepath=path_to_model)

    def predict(self, text: str) -> str:
        if MAX_LENGTH < len(text):
            logger.debug("Try to translate too long sentence")
            return dedent(f"""Your text has length {len(text)} > {MAX_LENGTH}.
            Please, reduce text for translator""")
        return self.model.predict(text)
