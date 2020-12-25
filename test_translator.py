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


@pytest.mark.slow
def test_opus_can_load_stored_model(tmpdir):
    print("Cur Dir ", os.path.dirname(__file__))
    model_filepath = os.sep.join([os.path.dirname(__file__), DEFAULT_DB_FILEPATH])
    print(f"Absolute path to model : {model_filepath}")
    model = OPUSModel(model_filepath)
    assert model.model.num_parameters() > 1000000 ,(
        "Model has no parameters, check the storage"
    )
    assert 62518 == model.tokenizer.vocab_size, (
        f"Vocabulary size does not compare with 62518, it equals {model.tokenizer.vocab_size}"
    )
    assert model.tokenizer("This is the pen.").input_ids == [268, 34, 4, 13155, 3, 0]
    assert 62518 == model.tokenizer.vocab_size
    assert 'Это моя ручка' in model.predict("This is my pen")
    # check the long text
    # assert ""

@pytest.mark.slow
def test_predictor_can_load_stored_model(tmpdir):
    # print("Cur Dir ", os.path.dirname(__file__))
    model_filepath = os.sep.join([os.path.dirname(__file__), DEFAULT_DB_FILEPATH])
    # print(f"Absolute path to model : {model_filepath}")
    # model = OPUSModel(model_filepath)
    predictor = CommonPredictor(OPUSModel, model_filepath)
    input_text = "I love to listen music."
    target_text = '<pad> Я люблю слушать музыку.'
    predicted_text = predictor.predict(input_text)
    assert predicted_text == target_text
    #
    # check the lenght error
    assert 'reduce text for' in predictor.predict(input_text * 100)


@pytest.mark.slow
def _test_from_pretrained_identifier():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING)
    assert isinstance(model, BertForMaskedLM)
    assert model.num_parameters() == 14410
    assert model.num_parameters(only_trainable=True) == 14410

