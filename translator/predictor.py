from textwrap import dedent

from translator.base_model import BaseModel
# from translator import OPUSModel

MAX_LENGTH = 250

class Predictor:

    def __init__(self, model, path_to_model: str):
        print(f"Load the model by path {path_to_model} or by pattern factory")

    def predict(self, text: str) -> str:
        return f"Return the translation of text: {text}"

    def evaluate(self, metric: str, *kwarg)->str:
        return "0"



class CommonPredictor(Predictor):
    """ Instance of OPUSModel transformer.

        Load from
    """
    def __init__(self, model: BaseModel, path_to_model: str = None):
        print(f"Init CommonPredictor with args:  ")
        self.model = model(tokenizer_filepath=path_to_model)

    def predict(self, text: str) -> str:
        if MAX_LENGTH < len(text):
            return dedent(f"""Your text has length {len(text)} > {MAX_LENGTH}.
            Please, reduce text for translator""")
        return self.model.predict(text)
