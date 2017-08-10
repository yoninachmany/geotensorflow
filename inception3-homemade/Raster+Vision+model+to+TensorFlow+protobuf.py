
# coding: utf-8

# # Get model from Raster Vision

# In[1]:


# 1. https://github.com/azavea/raster-vision/blob/develop/src/rastervision/run.py#L34-L35
# 2. https://github.com/azavea/raster-vision/blob/develop/src/rastervision/tagging/run.py#L23-L31
# 3. https://github.com/azavea/raster-vision/blob/develop/src/rastervision/common/run.py#L58-L68
# 4. https://github.com/azavea/raster-vision/blob/develop/src/rastervision/common/models/factory.py#L31-L43


# In[2]:


# https://github.com/azavea/raster-vision/blob/develop/src/rastervision/common/models/factory.py#L1
from os.path import isfile, join

# https://github.com/azavea/raster-vision/blob/develop/src/rastervision/common/models/factory.py#L31-L43
# Change: remove self, generator
def get_model(run_path, options, use_best=True):
    # Get a model by loading if it exists or making a new one.
    model_path = join(run_path, 'model.h5')

    # Load the model if it's saved, or create a new one.
    if isfile(model_path):
        model = load_model(run_path, options, use_best)
        print('Continuing training from saved model.')
    else:
        model = make_model(options)
        print('Creating new model.')

    return model


# In[3]:


# 5. https://github.com/azavea/raster-vision/blob/develop/src/rastervision/common/models/factory.py#L15-L29
# 6. https://github.com/azavea/raster-vision/blob/develop/src/rastervision/tagging/models/factory.py#L15-L50


# In[4]:


# https://github.com/azavea/raster-vision/blob/develop/src/rastervision/tagging/models/factory.py#L2-L8
from rastervision.common.models.resnet50 import ResNet50
from rastervision.common.models.densenet121 import DenseNet121
from rastervision.common.models.densenet169 import DenseNet169

BASELINE_RESNET = 'baseline_resnet'
DENSENET_121 = 'densenet121'
DENSENET_169 = 'densenet169'

# https://github.com/azavea/raster-vision/blob/develop/src/rastervision/tagging/models/factory.py#L15-L50
# Change: remove self, generator. Change dot notation to bracket notation (why didn't dot notation work?)
def make_model(options):
    """Make a new model."""
    model_type = options["model_type"]
    nb_channels = len(options["active_input_inds"])
    image_shape = (256, 256) # TODO: read in
    input_shape = (image_shape[0], image_shape[1], nb_channels)
    classes = 17 # TODO: read in
    
    weights = 'imagenet' if options["use_pretraining"] else None
    if model_type == BASELINE_RESNET:
        # A ResNet50 model with sigmoid activation and binary_crossentropy
        # as a loss function.
        model = ResNet50(
            include_top=True, weights=weights,
            input_shape=input_shape,
            classes=classes,
            activation='sigmoid')
    elif model_type == DENSENET_121:
        model = DenseNet121(weights=weights,
                            input_shape=input_shape,
                            classes=classes,
                            activation='sigmoid')
    elif model_type == DENSENET_169:
        model = DenseNet169(weights=weights,
                            input_shape=input_shape,
                            classes=classes,
                            activation='sigmoid')
    else:
        raise ValueError('{} is not a valid model_type'.format(model_type))

    if options.get("freeze_base", False):
        for layer in model.layers[:-1]:
            layer.trainable = False

    return model


# In[5]:


# https://github.com/azavea/raster-vision/blob/develop/src/rastervision/common/models/factory.py#L19-L29
# Change: remove self, generator
def load_model(run_path, options, use_best=True):
    #Load an existing model.
    # Load the model by weights. This permits loading weights from a saved
    # model into a model with a different architecture assuming the named
    # layers have compatible dimensions.
    model = make_model(options)
    file_name = 'best_model.h5' if use_best else 'model.h5'
    model_path = join(run_path, file_name)
    # TODO raise exception if model_path doesn't exist
    model.load_weights(model_path, by_name=True)
    return model


# In[6]:


import os
inception_test_augmentation_false_run_name = "tagging/7_18_17/inception/sgd"
tiff_pre_test_augmentation_run_name = "tagging/6_19_17/RGBtiff"
test_augmentation_run_name = "tagging/7_17_17/resnet_transform/0"
jpg_pre_test_augmentation_run_name = "tagging/7_3_17/baseline-branch-tiffdrop"
run_name = test_augmentation_run_name

raster_vision_data_path = os.environ.get("RASTER_VISION_DATA_DIRECTORY", None) or "/opt/data"
results_path = os.path.join(raster_vision_data_path, "results")
run_path = os.path.join(results_path, run_name)

import json
with open(os.path.join(run_path, "options.json")) as file:
    options = json.load(file)

# https://keras.io/backend/
from keras import backend as K
K.set_learning_phase(0) # test
model = load_model(run_path, options, use_best=True)
# to test with vanilla inceptionv3 model
# from keras.applications.inception_v3 import InceptionV3
# model = InceptionV3()

print(model.inputs)
print(model.outputs)


# In[7]:


output_node_tensor_name = model.outputs[0].name
output_node_name = output_node_tensor_name[0:output_node_tensor_name.index(":")]


# # Freeze model to protobuf

# In[8]:


# Background: https://www.tensorflow.org/extend/tool_developers/#freezing
# Template: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph_test.py#L39-L78
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib

def testFreezeGraph():
    temp_dir = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    checkpoint_prefix = os.path.join(temp_dir, "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "input_graph.pb"
    output_graph_name = "output_graph.pb"

    with K.get_session() as sess:
        saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
        checkpoint_path = saver.save(
            sess,
            checkpoint_prefix,
            global_step=0,
            latest_filename=checkpoint_state_name)
        graph_io.write_graph(sess.graph, temp_dir, input_graph_name)

    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(temp_dir, input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    output_node_names = output_node_name
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    # Only write the output graph to the run_path
    output_graph_path = os.path.join(run_path, output_graph_name)
    clear_devices = False

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_path, clear_devices, "")

    import shutil
    shutil.rmtree(temp_dir)


# In[9]:


testFreezeGraph()


# In[ ]:




