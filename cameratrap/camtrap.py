
from os import stat
import urllib
from pathlib import Path
from typing import List
from itertools import chain
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array

from .preprocessing.hoebeke import extract_boxes
from .preprocessing.megadetector import TFDetector
from .network.resnet_cam import object_localization as hoebeke_localization
from .network.resnet50_hierarchical_bottleneck_predict import predict_probabilities
from .network.hierachical_processing_predictions import probabilities_to_classification

RATIO = 0.5


@dataclass()
class CameraTrapImage():
    """Class for keeping track of a camera trap image, based on the camtrap-dp data format

    On first initialization of an image, the corresponding image will be
    downloaded to local disk. The image is stored as an Numpy ndarray as the ``data``
    attribute.
    """

    multimedia_id: str
    deployment_id: str
    sequence_id: str
    timestamp: datetime
    file_path: str
    file_name: str
    file_mediatype: str
    comments: str
    local_data_path: Path
    data: np.ndarray = field(init=False, repr=False)
    local_file_path: Path = field(init=False)
    _color_type: str = field(init=False)

    def __post_init__(self):
        self.local_file_path = self.local_data_path / self.deployment_id / self.file_name

        # Download if not yet available
        if not self.local_file_path.exists():
            print(f"{self.local_file_path} does not exist yet, downloading...")
            self.local_file_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.file_path, self.local_file_path)
            print(f"...download {self.local_file_path} succesful.")

        image = Image.open(self.local_file_path)

        # Preprocess image
        self._check_color(image)
        self.data = np.asarray(image)

    def crop(self, box):
        """Crop cameratrap image using using PIL crop method with given box

        Parameters
        ----------
        box : tuple (left, upper, right, lower)
            crop rectangle, as a (left, upper, right, lower)-tuple.

        Returns
        -------
        image : PIL Image
        """
        image = Image.fromarray(self.data)
        image = image.crop(box)
        return image

    @property
    def color_type(self):
        """Return color_type of the image."""
        return self._color_type

    def _check_color(self, image):
        """Check if incoming reconyx image is greyscale or color

        Parameters
        ----------
        image : PIL.Image
            Image

        Notes
        -----
        RGB and greyscale Reconyx images are both encoded with 3 channels and pillow reads them in RGB
        mode. Reconyx greyscale images have three channels the same values except of the logo, which has
        different values among the three layers. Hence, to check if greyscale, we compare the top half of
        image among the three channels. If equal, convert to greyscal image, otherwise set to RGB.
        """
        np_image = np.asarray(image)
        if np.all(np.diff(np_image[:, np_image.shape[1] // 2, :], axis=-1) == 0):
            self._color_type = 'grey'
        else:
            self._color_type = 'color'

    @classmethod
    def from_multimedia(cls, multimedia, multimedia_id, local_data_path):
        """Return CameraTrapImage from multimedia data table

        Parameters
        ----------
        mutimedia : pandas.DataFrame
            Multimedia data table conform the camtrap-dp format.
        multimedia_id : str
            Unique identifier of the multimedia item.
        local_data_path : pathlib.Path
            Main local data folder, containing the deployment(s) with the images as subfolder(s).
        """
        assert (isinstance(local_data_path, Path)), "local_data_path is not a valid Path."
        assert (multimedia_id in multimedia["multimedia_id"].values), "Multimedia ID not available in data."

        metadata = multimedia[multimedia["multimedia_id"] == multimedia_id].to_dict('records')[0]
        metadata['timestamp'] = pd.to_datetime(metadata['timestamp'])

        return cls(**metadata, local_data_path=local_data_path)

    def resize(self, ratio=RATIO, *args, **kwargs):
        """Resize Image using PIL resize method

        Parameters
        ----------
        ratio : float [0-1]
            Ratio to which the resizing should happen
        *args, **kwargs :
            Parameters routed to the PIL resize method
        """
        image = Image.fromarray(self.data)
        (width, height) = (int(image.width * ratio), int(image.height* ratio))
        image_resized = image.resize((width, height), *args, **kwargs)
        self.data = np.asarray(image_resized)

    def plot(self, ax=None):
        """Plot Image with matplotlib

        Parameters
        ----------
        ax : plt.axes
            Matplotlib axes to use as container to plot the image.
        """
        if not ax:
            fig, ax = plt.subplots()
        return ax.imshow(self.data)


@dataclass()
class CameraTrapSequence():
    """Class for keeping track of a camera trap image sequences, based on the camtrap-dp data format

    On first initialization of a sequence, the corresponding images of the sequence will be
    downloaded to local disk.
    """

    deployment_id: str
    sequence_id: str
    local_data_path: Path
    sequence: List[CameraTrapImage] = field(default_factory=list, repr=False)
    _color_type: str = field(init=False)

    def __post_init__(self):
        # Check all images in the sequence have the same color type
        assert len(set([img.color_type for img in self.sequence])) == 1, \
                "Images in sequence need to have same color type"
        self._color_type = self.sequence[0].color_type

        # remove the border
        self._crop_border()

    def __getitem__(self, idx):
        """Access an individual Image of the sequence"""
        return self.sequence[idx]

    @property
    def color_type(self):
        """Return color_type of the sequence."""
        return self._color_type

    @classmethod
    def from_multimedia(cls, multimedia, sequence_id, local_data_path):
        """Create Sequence setup from camtrap-db asset table data

        Parameters
        ----------
        multimedia : pandas.DataFrame
            Multimedia data table conform the camtrap-dp format.
        sequence_id : str
            Identifier of the sequence of photos.
        local_data_path : pathlib.Path
            Main local data folder, containing the deployment(s) with the images as subfolder(s).
        """
        assert (sequence_id in multimedia["sequence_id"].unique()), "Sequence ID not available in multimedia data."
        sequence_data = multimedia[multimedia["sequence_id"] == sequence_id]

        assert (len(sequence_data["deployment_id"].unique()) == 1), "Sequence not part of a single deployment."
        deployment_id = sequence_data["deployment_id"].unique()[0]

        sequence = []
        for row, multimedia in sequence_data.iterrows():
            multimedia_id = multimedia["multimedia_id"]
            sequence.append(CameraTrapImage.from_multimedia(sequence_data, multimedia_id, local_data_path))

        return cls(deployment_id, sequence_id, local_data_path, sequence)

    def _crop_border(self):
        """Remove the Reconyx black border on top and bottom of the image using background image

        Notes
        -----
        Check is done on the greyscale version of the first image of the deployment
        """
        reference_image = Image.fromarray(self.sequence[0].data)
        img_grey = np.asarray(reference_image.convert('L'))
        top, bottom = np.flatnonzero(img_grey[:, 0])[0], np.flatnonzero(img_grey[:, 0])[-1]
        for image in self.sequence:
            # TODO: check why Laura did -1
            image.data = np.asarray(image.crop((0, top, reference_image.size[0] - 1, bottom)))
        return image

    def resize(self, ratio=RATIO, *args, **kwargs):
        """Resize all images in the sequence

        Parameters
        ----------
        ratio : float [0-1]
            Ratio to resize the current images.
        *args, **kwargs :
            Additional parameters sent to the PIL.resize method.
        """
        for img in self.sequence:
            img.resize(ratio, *args, **kwargs)

    def draw_regions_of_interest(self, box_list, file_path="../data/interim/draw"):
        """Draw regions of interest on each of the images and save to disk

        Parameters
        ----------
        box_list : list
            A list with for each image a list with the different regions of interest. These
            regions of interest are defined as 4-tuple,
            where coordinates are (left, upper, right, lower; aka [x0, y0, x1, y1]).
        file_path : str | pathlib.Path
            Local file directory to store the output images with the regions of interest added.
        """
        assert len(box_list) == len(self.sequence), \
                "Length box_list need to equal number of images in sequence"
        for idx, image_ in enumerate(self.sequence):
            image = Image.fromarray(image_.data)
            im_box = ImageDraw.Draw(image)
            for box in box_list[idx]:
                if box:
                    im_box.rectangle(box, fill=None, outline='white')
            image.save(Path(file_path) / image_.file_name)

    def crop_regions_of_interest(self, box_list, file_path="../data/interim/crop"):
        """Crop regions of interest on each of the images and save to disk

        When multiple regions are derived from a single image, the images are defined
        by their file name + `-region_ID`.

        Parameters
        ----------
        box_list : list
            A list with for each image a list with the different regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        file_path : str | pathlib.Path
            Local file directory to store the output images with cropped regions
        """
        assert len(box_list) == len(self.sequence), \
                "Length box_list need to equal number of images in sequence"
        for image, region_list in zip(self, box_list):
            if region_list[0]:
                for idx, region in enumerate(region_list):
                    cropped = image.crop(region)
                    cropped_file_name = image.local_file_path.stem + f"-region_{idx}" + image.local_file_path.suffix
                    cropped.save( Path(file_path) / cropped_file_name)

    def detect(self, detection_algorithm, *args, **kwargs):
        """Detect the current ImageSequence species on image

        No classification, only bounding boxes on animals to define `regions_of_interest`.

        Parameters
        ----------
        detection_algorithm : subclass of BaseDetection
            Class can be custom defined, but needs to have a compliant ``detect`` method.
        *args, **kwargs :
            Additional arguments passed to the ``detect`` method of the detection_algorithm class

        Returns
        -------
        box_list : list with for each image in sequence a list of box-tuples
            Returns for each image in the sequence a list with the regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        """
        detector = detection_algorithm()
        return detector.detect(self.sequence, *args, **kwargs)

    def predict(self, classification_algorithm, box_list, *args, **kwargs):
        """Predict the current ImageSequence species observations

        The current implementation assumes a detection preprocessing step,
        but providing a box list of the original dimensions would

        Parameters
        ----------
        classification_algorithm : subclass of BaseClassification
            Class can be custom defined, but needs to have a compliant ``predict`` method.
        box_list : list with for each image in sequence a list of box-tuples
            Returns for each image in the sequence a list with the regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        *args, **kwargs :
            Additional arguments passed to the ``predict`` method of the classification_algorithm class

        Returns
        -------
        classification_algorithm ``predict`` returned output (can be algorithm dependent)
        """
        classification = classification_algorithm(*args, **kwargs)
        return classification.predict(self.sequence, box_list)


class BaseDetection():
    """Base class for detection algorithms of species on images"""

    def detect(self, sequence):
        """Overwrite detection of regions of interest in inherited subclasses"""
        NotImplemented

class DetectionDeactive():
    """Ignore detection by returning bounding boxes that are equal to the original images"""

    def detect(self, sequence):
        """Dummy region of interest extracting the image dimensions itself

        Parameters
        ----------
        sequence : CameraTrapSequence.sequence
            Sequence (List[CameraTrapImage]) of images.
        """
        box_list = []
        for image in sequence:
            height, width, _ = image.data.shape
            box_list.append((0, 0, width, height))
        return box_list

class DetectionHoebeke():
    """Detection algorithm by Hoebeke L."""

    @staticmethod
    def _calculate_median_background(sequence):
        """Calculate the background as the median of all images in the sequence"""
        dim = sequence[0].data.ndim
        image_stack = np.concatenate([img.data[..., None] for img in sequence], axis=dim)
        median_array = np.median(image_stack, axis=dim)
        median_image = Image.fromarray(median_array.astype('uint8'))
        return median_image

    def detect(self, sequence):
        """Region of interest detection algorithm by method L. Hoebeke

        Parameters
        ----------
        sequence : CameraTrapSequence.sequence
            Sequence (List[CameraTrapImage]) of images.

        Returns
        -------
        box_list : list with for each image in sequence a list of box-tuples
            Returns for each image in the sequence a list with the regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        """
        median = self._calculate_median_background(sequence)

        box_list = []
        for image in sequence:
            box, _ = extract_boxes(Image.fromarray(image.data), median)
            box_list.append(box)
        return box_list


class DetectionMegaDetector():
    """Region of interest detection algorithm by https://github.com/microsoft/CameraTraps"""

    def __init__(self):
        self.detector = TFDetector("../cameratrap/weights/md_v4.1.0.pb")

    def detect(self, sequence):
        """Extract the regions of interest according Megadetector model

        Parameters
        ----------
        sequence : CameraTrapSequence.sequence
            Sequence (List[CameraTrapImage]) of images.

        Returns
        -------
        box_list : list with for each image in sequence a list of box-tuples
            Returns for each image in the sequence a list with the regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        """
        box_list = []
        for image in sequence:
            megadetector_output = self.detector.generate_detections_one_image(
                Image.fromarray(image.data), image.multimedia_id)

            # only return animal IDs as bbox to conform
            # TODO USE INFO on confidence level?
            box = []
            for detection in megadetector_output["detections"]:
                animal_id = list(
                    TFDetector.DEFAULT_DETECTOR_LABEL_MAP.keys())[
                        list(TFDetector.DEFAULT_DETECTOR_LABEL_MAP.values()).index("animal")
                    ]
                if detection["category"] == animal_id:
                    box.append(detection["bbox"])
            box_list.append(box)
        return box_list


class BaseClassification():
    """Base class for classification algorithms of species on images"""

    def predict(self, sequence):
        """Overwrite prediction of classificiation classes in inherited subclasses

        Parameters
        ----------
        sequence : CameraTrapSequence.sequence
            Sequence (List[CameraTrapImage]) of images.
        """
        NotImplemented

    def localize(sequence, box_list, file_path="../data/interim/localize"):
        """Overwrite localization of the models in inherited subclasses

        Parameters
        ----------
        box_list : list
            A list with for each image a list with the different regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        file_path : str
            File path to write the output images.
        """
        NotImplemented

class ClassificationTensorflow():

    def __init__(self, model_path, class_labels=None):
        """Use Tensorflow model to make classification prediction

        Parameters
        ----------
        model_path : path of tensorflow model
            Tensorflow model main path
        """
        self.model = tf.saved_model.load(model_path)
        self.class_labels = class_labels

    def _predict_single_image(self, image):
        """Predict single image

        Parameters
        ----------
        image : pillow.Image
            Image to create prediction on

        See also
        --------
        https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
        """
        image = np.asarray(image)
        # Model input needs to be a tensor
        input_tensor = tf.convert_to_tensor(image)
        # TModel expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy()
                    for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        return output_dict

    def predict(self, sequence, box_list):
        """Predict the current sequence output

        Parameters
        ----------
        sequence : ImageSequence
            Sequence of images
        box_list : list
            A list with for each image a list with the different regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        class_labels : dict {idx : label,...}
            Dictionary containing the class labels corresponding to the model output identifiers

        Notes
        -----
        Each image is run individually, as we have no guarantee all oxes have equal size
        # TODO - give input equal size
        """
        predictions = []
        for idx, image_ in enumerate(sequence):
            for box in box_list[idx]:
                if box:
                    predictions.append(self._predict_single_image(image_.crop(box)))

        # Add labels if these are available:
        if self.class_labels:
            for prediction in predictions:
                prediction['detection_classes_names'] = [self.class_labels[idx]
                    for idx in prediction['detection_classes']]
        return predictions


class ClassificationHoebeke():

    def __init__(self, model_path):
        self.model_file_path = model_path

    def box_counter(self, box_list):
        """Count the number of boxes in a list of list of 4-tuple boxes"""
        total_boxes = 0
        for el in chain([box for box in box_list if box[0]]):
            total_boxes += len(el)
        return total_boxes

    def _prepare_image_matrix(self, sequence, box_list):
        """Combine all boxes into a single 4D numpy array

        Parameters
        ----------
        sequence : ImageSequence
            Sequence of images
        box_list : list
            A list with for each image a list with the different regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        """
        # Image size (currently hardcoded as L. Hoebeke)
        dim_x = 270
        dim_y = 480
        dim_z = 3

        total_boxes = self.box_counter(box_list)
        batch_images = np.zeros((total_boxes, dim_x, dim_y, dim_z))

        for idx, image_ in enumerate(sequence):
            for i, box in enumerate(box_list[idx]):
                if box:
                    cropped_box = img_to_array(image_.crop(box))
                    cropped_box = np.expand_dims(cropped_box, axis=0)
                    image_prepared = preprocess_input(np.copy(cropped_box),
                                                      data_format=None)
                    batch_images[i] = image_prepared

        return batch_images

    def predict_bottleneck_features(self, sequence, box_list):
        """Calculate the bottleneck features a set of regions of interest defined as box.

        Parameters
        ----------
        sequence : CameraTrapSequence.sequence
            Sequence (List[CameraTrapImage]) of images.
        box_list : list
            A list with for each image a list with the different regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        """
        assert len(box_list) == len(sequence), "Length box_list need to equal number of images in sequence"

        # Prepare the cropped images
        batch_images = self._prepare_image_matrix(sequence, box_list)

        # Build the ResNet50 network
        model = ResNet50(include_top=False, weights='imagenet',
                         input_shape=(270, 480, 3))  # Hardcoded as done by L. Hoebeke

        # Add an additional Pooling layer to retrofit previous edition L. Hoebeke
        x = model.output
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        my_model = Model(inputs=model.input, outputs=x)

        bottleneck_features = my_model.predict(batch_images)

        return bottleneck_features

    @staticmethod
    def _predict_hierarchical_classes(bottleneck_features, file_path_weights):
        """Calculate the probabilities for each class"""
        return predict_probabilities(bottleneck_features, file_path_weights)

    @staticmethod
    def _hierarchical_predictions_sequences(predictions):
        """Convert the probabilities to the hierarchical classes"""
        return probabilities_to_classification(predictions)

    @staticmethod
    def localize(sequence, box_list, file_path="../data/interim/localize"):
        """Check which regions are activated during prediction

        Parameters
        ----------
        box_list : list
            A list with for each image a list with the different regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        file_path : str, default '../data/interim/localize'
            File path to write the output images.
        """
        assert len(box_list) == len(sequence), \
                "Length box_list need to equal number of images in sequence"
        # create localize dir if not existing
        Path(file_path).mkdir(parents=True, exist_ok=True)

        for image, region_list in zip(sequence, box_list):
            if region_list[0]:
                for idx, region in enumerate(region_list):
                    cropped = image.crop(region)
                    cropped_file_name = image.local_file_path.stem + f"-region_{idx}" + image.local_file_path.suffix
                    cropped.save( Path(file_path) / cropped_file_name)

                    fig, ax = plt.subplots()
                    ax = hoebeke_localization(str(Path(file_path).resolve() / cropped_file_name), ax=ax)
                    fig.savefig(Path(file_path) / cropped_file_name)

    def predict(self, sequence, box_list):
        """Predict the current sequence output

        Parameters
        ----------
        sequence : ImageSequence
            Sequence of images
        box_list : list
            A list with for each image a list with the different regions of interest. These
            regions of interest are defined as 4-tuple, where coordinates are (left, upper, right, lower).
        """
        bottleneck_features = self.predict_bottleneck_features(sequence, box_list)
        predictions = self._predict_hierarchical_classes(bottleneck_features, self.model_file_path)
        return self._hierarchical_predictions_sequences(predictions)

