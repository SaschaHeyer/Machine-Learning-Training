gcloud ai-platform local train \
--package-path trainer \
--module-name trainer.task \
--job-dir local-training-output \
-- \
--train-file ./data/ner.csv