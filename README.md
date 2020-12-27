# translator

[архив модели OPUS-mt-en-ru](https://drive.google.com/file/d/1rPQ7s9L9Yx0w1ZAHY74zsR-1umANKcu7/view?usp=sharing)

Скачать , разархивировать в директорию `data`

Описание модуля `translator` (translator/README.md), В нем же приведено описание моделей и методов оценки модели.


В самом простом варианте реализация переводчика занимает несколько строк:

```python
from translator.src.predictor import CommonPredictor
from translator.src.translator import OPUSModel

predictor = CommonPredictor(OPUSModel, './data/stored-opus-mt-en-ru')

predicted_text = predictor.predict("My name is Nikolai and I live in Almetievsk.")

print(f"Translation: {predicted_text}")
```
