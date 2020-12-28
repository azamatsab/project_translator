from flask import Flask, request

from translator.src.predictor import CommonPredictor
from translator.src.translator import OPUSModel

import json


SUPPORTED_LANGUAGE_PAIRS = {
    ('EN', 'RU')
}


def create_app():
    new_app = Flask(__name__)
    new_app.config.from_object('config')
    return new_app


app = create_app()

predictor = CommonPredictor(OPUSModel, '../data/stored-opus-mt-en-ru')


@app.route('/v1/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        json_str = request.data.decode('utf-8')
        data = json.loads(json_str)
        source_lang = data['source_lang']
        target_lang = data['target_lang']
        original = data['text']
        is_supported = (source_lang, target_lang) in SUPPORTED_LANGUAGE_PAIRS
        if is_supported:
            response = {
                'translated_text': predictor.predict(original) if original else "",
                'status': 'success'
            }
        else:
            response = {
                'comment': f'Unsupported language pair {source_lang} -> {target_lang}',
                'status': 'error'
            }, 406
        return response
    return ''


@app.route('/')
def root():
    return 'Welcome!'


if __name__ == '__main__':
    # TODO: consider moving this to configs
    app.run(debug=True, port=3824)
