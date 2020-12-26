
class Predictor:

    def __init__(self, model, path_to_model: str):
        print(f"Load the model by path {path_to_model} or by pattern factory")

    def predict(self, text: str) -> str:
        return f"Return the translation of text: {text}"

    def evaluate(self, metric: str, *kwarg)->str:
        return "0"

