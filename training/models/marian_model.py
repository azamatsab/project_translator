from transformers import MarianMTModel, MarianTokenizer

from training.models.base_model import BaseModel

TOKENIZER_PATH = "Helsinki-NLP/opus-mt-en-ru"

class Marian(BaseModel):
	def __init__(self, tokenizer_path: str):
		self.tokenizer = MarianTokenizer.from_pretraine(tokenizer_path)

