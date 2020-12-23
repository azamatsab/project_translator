class Predictor:

    def __init__(self, path_to_model: str):
        print(f"Load the model by path or by pattern factory")

    def predict(self, text: str) -> str:
        return f"Return the translation of text: {text}"

    def evaluate(self, metric: str, …)->str:
        return "0"
