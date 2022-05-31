
# Cameratrap Image recognition

## Introduction

This repo contains the code to process camera trap images and apply the prediction model developed by L Hoebeke.

## Code development

The setup is inspired by the [data science cookiecutter](https://drivendata.github.io/cookiecutter-data-science/) providing a
consistent structure. Please make sure to read the section on [notebooks](https://drivendata.github.io/cookiecutter-data-science/#notebooks-are-for-exploration-and-communication).
We aim to have as much reusable code in the `cameratrap` directory as possible (cfr. lego blocks to play around with), grouped by topic in specific files.

In order to make it possible to develop functions and use them directly in a notebook, the following elements are important:

- Setup/update the environment: Dependencies are collected in the conda `environment.yml` file, so anybody can recreate the required environment using:

```
conda env create -f environment.yml
conda activate py38-cameratrap-t2
```

- Install the code developed specifically for the project (lives inside the `cameratrap` folder) in the environment (in `-e` edit mode):

```
conda activate py38-cameratrap-t2
pip install -e .
```

- When developing the code in the `cameratrap` folder, make sure you can directly try it in a running notebook (you did read [notebooks section](https://drivendata.github.io/cookiecutter-data-science/#notebooks-are-for-exploration-and-communication)) already?

```
# OPTIONAL: Load the "autoreload" extension so that code can change
%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
%autoreload 2

from cameratrap import CameraTrapImage, CameraTrapSequence
```

__Note:__ The setup of Jupyter notebook/lab is not part of the environment and should be setup separately, either inside the environment itself (`conda activate env-cameratrap;conda install jupyterlab`) or using [`nb_conda_kernels`](https://github.com/Anaconda-Platform/nb_conda_kernels). The latter let you access different conda environments from within a single jupyter lab installation.

## External models

The implementation depends on some external models, which should not be part of version control, but need to be downloaded and put in the `weights` folder:
- [`weights/md_v4.1.0.pb`](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md#downloading-the-models)  -> Microsoft megadetector algortihm to extract animals from images
- [resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5](https://github.com/fchollet/deep-learning-models/releases/tag/v0.2) -> resnet base model on which Hoebeke build classification implementation

## Data

The data format used to describe the assets (multimedia images), observations and metadata of a camera trap project is the [camptrap-db](https://gitlab.com/oscf/camtrap-dp) format, which is a specialized defintion of the [frictionless data package](https://frictionlessdata.io/) for camera trap projects.

See the tutorial notebook `tutorial_predictions.ipynb` to load the data from a camtrap-db data package.

### Pretrained-models

- For the detection tutorial, the [MegaDetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md#downloading-the-model) model is used to detect regions of interest (bounding boxes). The model file is too large to be included in the repository, so make sure to [download the model](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb) first and put it in the `weights` subfolder, i.e. `cameratrap/weights/md_v4.1.0.pb`.

- For the classification tutorial, the [inaturalist species trained model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#inaturalist-species-trained-models) which was trained on the iNaturalist data set is used as example. Before running the tutorial, make sure to download the model, extract it and save it into the weights folder. The model directory should be `cameratrap/weights/faster_rcnn_resnet101_fgvc_2018_07_19/saved_model`.

To apply these algorithms on other pretrained models, one can use a similar approach.

### Data inside repository

As most of the data is too large to contain inside in a github repository, content of the data-folder is ignored by git. The assumed directory structure for the cameratrap data is as follows:

```
data
├── interim                                     # interim subfolders are setup by the application code
│   ├── ...
│   └── draw
├── ...
└── raw
    ├── 05c05f7b-6c38-4a06-a3b0-145de226c8ad    # deployment downloaded for local development
    ├── 79c4317b-0df9-4c86-a2df-4557cde3345d
    ├── ...                                     # more locally downloaded deployments
    ├── datapackage.json                        # metadata and schemas for the frictionless data package
    ├── deployments.csv                         # other camtrap-db data package files
    ├── multimedia.csv
    └── observations.csv
```

### Example data for MICA project

Some `sequence_ids` useful to test with:

- Muskusratten:
    - 9cdf72bd-483a-4a5d-acd9-5e137964a14b
    - 50ce5d0c-8551-4d8f-a8c6-c6ed604635ee
    - 8271d989-5f1d-4ac7-858b-c8384c81cd35
    - d44384b4-36c8-41c4-9d05-d6efbbfb9905
    - 6e74c3ae-7de9-4816-b2cc-c14ff2698b90
- Beverratten:
    - c407758c-b7de-432d-9b7f-c905c4893722
    - 73408a43-dc09-4076-9a60-a7060fa38493
    - bf10a0f1-ba69-4030-adf8-fec535a6206a
    - 79747a30-f23b-496b-8a6c-99ad5082a95f
    - 18730b56-2e61-4468-b621-ac1a47a34330
- Bruine ratten:
    - 1f39943e-328f-4e1d-9f87-a0cccb376ffc
    - 6b66c8d2-1900-42d7-bb59-7379b1f4148f
    - 681971ee-6899-49d4-8ffa-eae0f671377b
- Bevers:
    - 068ce453-2a82-4969-89e6-62714d214592
    - 78c5c934-3dbb-4a43-833c-0a109247db7c
    - 1e9aca5c-68cf-46e6-b842-246e44779bcd