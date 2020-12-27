###### Для запуска обучения выполните скрипт training/pipeline.py
		python3 training/pipeline.py --model "model_name" --stage "stage" --dataset "opus" --dataset_path "path/to/dataset/folder" --config "path"

--model - имя модели, например "Helsinki-NLP/opus-mt-en-ru" или "t5-small"
--stage - этапы, с которого нужно начать обучение. Возможные значения: download, preprocess, dataset
--dataset - на данный момент поддерживается только opus
--dataset_path - путь в папку для загрузки датасета, если stage="download", и для чтения при создании датасетов
--config - путь к файлу конфигурации, пример в training/config.yaml

На данный момент trainer поддерживает только cpu и gpu. Вид устройства указывается в config.yaml

###### При каждом запуске обучения параметры и метрики логируются mlflow. Для просмотра логов запустите:
		mlflow ui и перейдите по ссылке