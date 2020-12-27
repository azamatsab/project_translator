###### Для запуска обучения выполните скрипт training/pipeline.py
		python3 training/pipeline.py --model "model_name" --stage "stage" --dataset "opus" --dataset_path "path/to/dataset/folder" --config "path"
Например:

		python3 training/pipeline.py --model "t5-small" --stage download --dataset OPUS --dataset_path dataset/ --path_to_yaml_params 	training/config.yaml

При таком запуске сначала будет скачан датасет opus в директорию dataset, дальше начнется обучение модели "t5-small", используя гиперпараметры из training/config.yaml
		
		python3 training/pipeline.py --model "Helsinki-NLP/opus-mt-en-ru" --stage dataset --dataset OPUS --dataset_path dataset/ --path_to_yaml_params training/config.yaml
В этом случае датасет будет загружен из директории dataset и начнется обучение модели "Helsinki-NLP/opus-mt-en-ru"

--model - имя модели, например "Helsinki-NLP/opus-mt-en-ru" или "t5-small"
--stage - этапы, с которого нужно начать обучение. Возможные значения: download, preprocess, dataset
--dataset - на данный момент поддерживается только opus
--dataset_path - путь в папку для загрузки датасета, если stage="download", и для чтения при создании датасетов
--config - путь к файлу конфигурации, пример в training/config.yaml

На данный момент trainer поддерживает только cpu и gpu. Вид устройства указывается в config.yaml

###### При каждом запуске обучения параметры и метрики логируются mlflow. Для просмотра логов запустите:
		mlflow ui и перейдите по ссылке