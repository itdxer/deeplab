import os
from functools import partial

import tensorflow as tf

from neupy import architectures, storage
from neupy.utils import as_tuple
from neupy.layers import *

from utils import download_file


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
RESNET50_WEIGHTS_FILE = os.path.join(CURRENT_DIR, 'models', 'resnet50.hdf5')
RESNET50_WEIGHTS_URL = "http://neupy.s3.amazonaws.com/tensorflow/imagenet-models/resnet50.hdf5"


class ResizeBilinear(BaseLayer):
    def __init__(self, new_shape=(None, None), *args, **kwargs):
        self.new_shape = new_shape
        super(ResizeBilinear, self).__init__(*args, **kwargs)

    @property
    def output_shape(self):
        if not self.input_shape:
            return

        elif None not in self.new_shape:
            return as_tuple(self.new_shape, self.input_shape[0][-1])

        elif len(self.input_shape) > 1:
            height, width, _ = self.input_shape[1]
            return as_tuple(height, width, self.input_shape[0][-1])

    def output(self, inputs):
        if None in self.new_shape:
            input_value, related_shape = inputs
            new_shape = tf.shape(related_shape)
            new_shape = (new_shape[1], new_shape[2])
        else:
            new_shape = self.new_shape
            input_value = inputs[0]

        return tf.image.resize_bilinear(input_value, new_shape)


class IncludeResidualInputs(BaseLayer):
    def __init__(self, *layers, **kwargs):
        self.layers = layers
        super(IncludeResidualInputs, self).__init__(**kwargs)

    def initialize(self):
        for layer in self.layers:
            self.graph.connect_layers(layer, self)

    def output(self, *inputs):
        return inputs


def patch_layers(network, patches):
    for layer in network:
        layer_class = layer.__class__

        if layer_class in patches:
            patch = patches[layer_class]

            for key, value in patch.items():
                setattr(layer, key, value)


def download_resnet50_weights():
    if not os.path.exists(RESNET50_WEIGHTS_FILE):
        download_file(
            url=RESNET50_WEIGHTS_UR,
            filepath=RESNET50_WEIGHTS_FILE,
            description='Downloading weights')

    print("File with ResNet-50 weights: {}".format(RESNET50_WEIGHTS_FILE))
    return RESNET50_WEIGHTS_FILE


def create_deeplab_model(resnet50_weights=None, deeplab_weights=None, size=None):
    print("Initializing ResNet-50 architecture...")

    SamePadConv = partial(Convolution, bias=None, padding='same')
    resnet50 = architectures.resnet50(
        input_shape=(size, size, 3),
        include_global_pool=False,
        in_out_ratio=16,
    )

    if resnet50_weights is not None:
        # Pre-trained ResNet-50 contains parameters for the final
        # classification layer. We don't use this layer and for this reason
        # we need to set ``ignore_missing=True``
        print("Recovering ResNet-50 parameters...")
        storage.load(resnet50, resnet50_weights, ignore_missing=True)

    in_height, in_width, _ = resnet50.input_shape
    out_height, out_width, _ = resnet50.output_shape

    resnet50_input = resnet50.layers[0]
    deeplab_input = Input(resnet50.output_shape, name='deeplab-input')

    print("Initializing Deeplab architecture...")
    deeplab = join(
        deeplab_input,

        # Atrous Spatial Pyramid Pooling
        parallel(
            SamePadConv((1, 1, 256)) > BatchNorm(),
            SamePadConv((3, 3, 256), dilation=6) > BatchNorm(),
            SamePadConv((3, 3, 256), dilation=12) > BatchNorm(),
            SamePadConv((3, 3, 256), dilation=18) > BatchNorm(),
            [
                GlobalPooling('avg'),
                Reshape((1, 1, -1)),
                SamePadConv((1, 1, 256)) > BatchNorm(),

                IncludeResidualInputs(deeplab_input),
                ResizeBilinear(),
            ]
        ),
        Concatenate(),
        SamePadConv((1, 1, 256)) > BatchNorm(),

        # Convert to the classification maps
        Convolution((1, 1, 21), padding='same'),
        IncludeResidualInputs(resnet50_input),
        ResizeBilinear((in_height, in_width)),
        Softmax(name='segmentation-proba'),
    )

    if deeplab_weights is not None:
        print("Recovering Deeplab parameters...")
        storage.load(deeplab, deeplab_weights, ignore_missing=True)

    print("Patching layers...")
    patches = {
        BatchNorm: {
            'alpha': 1 - 0.997,
            'epsion': 1e-5,
        }
    }
    patch_layers(deeplab, patches)
    patch_layers(resnet50, patches)

    return resnet50, deeplab
