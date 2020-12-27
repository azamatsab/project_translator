from argparse import Namespace
import os
import sys
from textwrap import dedent
from getpass import getpass
from unittest.mock import patch, MagicMock
import json
from json import JSONDecodeError
from contextlib import nullcontext as do_not_raise

import pytest
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


from translator.predictor import Predictor, CommonPredictor
from translator.translator import OPUSModel

DEFAULT_DB_FILEPATH = './data/stored-opus-mt-en-ru'
INPUT_TEXT = "I love to listen music."

@pytest.fixture(scope='session')
def translator_instance():
    model_filepath = os.sep.join([os.path.dirname(__file__), DEFAULT_DB_FILEPATH])
    model = OPUSModel(model_filepath)
    return model

@pytest.fixture(scope='session')
def predictor_instance():
    model_filepath = os.sep.join([os.path.dirname(__file__), DEFAULT_DB_FILEPATH])
    model = CommonPredictor(OPUSModel, model_filepath)
    return model

# @pytest.mark.skip
def test_tranlator_can_load_stored_model(translator_instance):
    assert translator_instance.model.num_parameters() > 1000000 ,(
        "Model has no parameters, check the storage"
    )

# @pytest.mark.skip
def test_tranlator_check_vocab_len(translator_instance):
    assert 62518 == translator_instance.tokenizer.vocab_size, (
        f"Vocabulary size does not compare with 62518, it equals {model.tokenizer.vocab_size}"
    )

# @pytest.mark.skip
def test_tranlator_check_tokenizer(translator_instance):
    assert translator_instance.tokenizer("This is the pen.").input_ids == [268, 34, 4, 13155, 3, 0]

# @pytest.mark.skip
def test_tranlator_check_predict(translator_instance):
    assert 'Это моя ручка' in translator_instance.predict("This is my pen")

# @pytest.mark.skip
def test_translator_check_input_len(translator_instance):
    assert 'reduce text for' in translator_instance.predict(INPUT_TEXT * 100)

def test_translator_can_save_and_load_model(caplog, tmpdir, translator_instance):
    translator_instance.save_model(tmpdir)
    pred = CommonPredictor(OPUSModel, tmpdir)
    target_text = '<pad> Я люблю слушать музыку.'
    predicted_text = pred.predict(INPUT_TEXT)
    assert predicted_text == target_text

# @pytest.mark.skip
def test_predictor_can_load_stored_model(predictor_instance):
    input_text = "I love to listen music."
    target_text = '<pad> Я люблю слушать музыку.'
    predicted_text = predictor_instance.predict(input_text)
    assert predicted_text == target_text

def test_predictor_check_input_len(predictor_instance):
    assert 'reduce text for' in predictor_instance.predict(INPUT_TEXT * 100)


@pytest.mark.slow
def _test_from_pretrained_identifier():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING)
    assert isinstance(model, BertForMaskedLM)
    assert model.num_parameters() == 14410
    assert model.num_parameters(only_trainable=True) == 14410

