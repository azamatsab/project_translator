from pathlib import Path


WEIGHTS_DIR = Path(__file__).resolve() / 'weights'

OPUS_LINKS = {"dev_en": "http://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-dev.en",
               "dev_ru": "http://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-dev.ru",
               "test_en": "http://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-test.en",
               "test_ru": "http://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-test.ru",
               "train_en": "http://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-train.en",
               "train_ru": "http://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-train.ru"
               }

OPUS_PATHS = {"train_en": "opus.en-ru-train.en",
			  "train_ru": "opus.en-ru-train.ru",
			  "test_en": "opus.en-ru-test.en",
			  "test_ru": "opus.en-ru-test.ru",
			  "dev_en": "opus.en-ru-dev.en",
			  "dev_ru": "opus.en-ru-dev.ru"
				}