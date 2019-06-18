# [WIP] Named Entity Recognition with Kubeflow and Keras 

This example demonstrates how you can use Kubeflow to train and deploy a Keras model with a custom prediction routine. 

Custom prediction routines allow you to determine what code runs when you send an online prediction request to AI Platform.

https://cloud.google.com/ml-engine/docs/tensorflow/custom-prediction-routines

Our model directory includes a HDF5 Keras model with a custom preprocessor. 

## Goals

* Demonstrate how to use Keras only models
* Demonstrate how to train a Named Entity Recognition model
* Demonstrate how to deploy a Keras model to AI Platform
* Demonstrate how to use a custom prediction routine
* Demonstrate how to use serve the model with AI Platform
* Demonstrate how to use Kubeflow metrics
* Demonstrate how to use Kubeflow visualizations 

## Components

This Kubeflow project contains 3 components

* preprocess
* train
* deploy

### Usage

1.  Build and push the Docker images for each component. Please change the `image_name` in the `build_image.sh` accordingly to your Google Container Registry. Do this step for all three components.

1. Upload the component specification (`component.yaml`) for all components to a Bucket. Change the `image` path in the specification to your Docker repository. 

1. Open the [NER-Pipeline.ipynb](NER-Pipeline.ibynp) in your Kubeflow environment, change the parameter according to your project and run it. 

1. Upload the dataset to a Bucket, the dataset can be downloaded [here](https://drive.google.com/file/d/136CqAq6z69ztIFCdswJl_CP7K3fddPn1/view?usp=sharing
) 

1. Upload the custom prediction routine [WIP]

## TODO
- [ ] Add the custom prediction routine implementation


