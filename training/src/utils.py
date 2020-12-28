import os
import yaml

DEFAULT_CONFIG_FILEPATH = os.sep.join(
    [
        os.path.dirname(__file__),
        '..',
        'constants.yml',
    ]
)

def get_configs():
    with open(DEFAULT_CONFIG_FILEPATH, 'r') as fin:
        cfg = yaml.safe_load(fin)
    return cfg