image_name=gcr.io/spielwiese-sascha-heyer/kubeflow/training/deploy-custom-prediction-routine
image_tag=latest

full_image_name=${image_name}:${image_tag}

docker build -t "${full_image_name}" .
docker push "$full_image_name"