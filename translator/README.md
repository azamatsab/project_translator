# Translator

Модуль `translator`

Написали класс `BaseModel` для моделей переводчиков, чтобы можно было Приведенную модель можно расширять для других моделей. В частности, внедрена модель на основе `MarianTokenizer` и `MarianMT` с обучением модели на наборе [OPUS-mt-en-ru](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru).


В репозитории [`huggingface/transformers`](https://github.com/huggingface/transformers) есть описание как использовать модели HF Seq2Seq для тренировки и оценки. 

Для подобных задач используются метрики BLEU и ROUGE в различных представлениях (ROUGE-L, ROUGE-N и др.). 

В документации отмечено:

    At the moment, Seq2SeqTrainer does not support with teacher distillation.

То есть при оптимизации модели методом `Seq2SeqTrainer` нельзя проводить уменьшение модели.

