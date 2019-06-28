JOB_NAME="ner_$(date '+%Y_%m_%d_%H_%M_%S')"
JOB_DIR=gs://sascha-ml-engine/models/ner

TRAIN_FILE=gs://sascha-ml-engine/data/ner/ner.csv

REGION=europe-west1

gcloud config set project machine-learning-sascha

gcloud ai-platform jobs submit training $JOB_NAME \
  --package-path trainer/ \
  --module-name trainer.task \
  --region $REGION \
  --config hyperparameter.yaml \
  --python-version 3.5 \
  --runtime-version 1.13 \
  --job-dir $JOB_DIR \
  --stream-logs \
  --verbosity=debug \
  -- \
  --train-file $TRAIN_FILE

  