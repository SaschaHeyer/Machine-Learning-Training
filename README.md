# [WIP] Machine-Learning-Training

> Work in progress

## Welcome to this repository
You might ask yourself what can I find here. This repository mainly contains different small and large projects related to machine learning. 

Contains jupyter notebooks and examples for Keras, Google AI Platform and Kubeflow. 

I update this repository while learning new technologies and topics.  **It is used as a self reminder for my future self.**

## Repository structure
    .
    ├── ai-platform                     # Implementations related to Google AI Platform.
    ├── kubeflow                        # Implementations related to kubeflow.
        ├── reusable-component-training # Explains the basics of components and pipelines in Kubeflow
        └── named-entity-recognition    # NER based on Keras and Kubeflow with AI Platform
    ├── notebooks                       # Several notebooks for different kind of technologies.
    ├── notes                           # Personal machine learning notes, as self reminder =).
    └── presentations                   # Personal presentations related to ml.

## Changelog

### v0.4
2019-06-26

* added training component for concatenating strings
* added training notebook for reusable components and pipelines

### v0.3
2019-06-18

* NER added deployment for AI Platform using beta feature for custom processing routine

### v0.2
2019-06-17

* NER added processing state export for custom prediction routine usage
* NER added logic to export keras model

### v0.1
2019-06-14

* Initial repository and structure.
* Added several basic kubeflow notebooks.
