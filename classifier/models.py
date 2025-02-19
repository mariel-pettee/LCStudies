import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.wrappers.scikit_learn import KerasRegressor
import sonnet as snt
from graph_nets import blocks
from graph_nets import modules

# Custom layers.
from layers import *


def make_mlp_model(n_layers = 2, dropout=0.1):
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  # the activation function choices:
  # swish, relu, relu6, leaky_relu
  return snt.Sequential([
      snt.nets.MLP([128, 64]*n_layers,
#     snt.nets.MLP([128, 64],
        activation=tf.nn.relu,
        activate_final=True,
#         dropout_rate=dropout
    ),
    snt.LayerNorm(axis=-1, create_scale=True, create_offset=False)
  ])

class MLPGraphNetwork(snt.Module):
    """GraphIndependent with MLP edge, node, and global models."""
    def __init__(self, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        self._network = modules.GraphNetwork(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=make_mlp_model
            )

    def __call__(self, inputs,
            edge_model_kwargs=None,
            node_model_kwargs=None,
            global_model_kwargs=None):
        return self._network(inputs,
                      edge_model_kwargs=edge_model_kwargs,
                      node_model_kwargs=node_model_kwargs,
                      global_model_kwargs=global_model_kwargs)
    
LATENT_SIZE = 128
NUM_LAYERS = 3

class GlobalClassifier(snt.Module):

    def __init__(self, name="GlobalClassifier"):
        super(GlobalClassifier, self).__init__(name=name)

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=make_mlp_model,
            use_edges = True,
            use_receiver_nodes = True,
            use_sender_nodes = True,
            use_globals = True,
            name='edge_encoder_block'
        )

        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=make_mlp_model,
            use_received_edges=True,
            use_sent_edges=True,
            use_nodes=True,
            use_globals=True,
            name='node_encoder_block'
        )

        self._global_block = blocks.GlobalBlock(
            global_model_fn=make_mlp_model,
            use_edges=True,
            use_nodes=True,
            use_globals=True,
        )

        self._core = MLPGraphNetwork()
        # Transforms the outputs into appropriate shapes.
        global_output_size = 1
        global_fn = lambda: snt.Sequential([
#             snt.nets.MLP([32, 64],),
#             snt.nets.MLP([128, 64],),
            snt.nets.MLP([128, 64, 128, global_output_size], name='global_output'),
            tf.sigmoid
        ])

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._global_block(self._edge_block(self._node_encoder_block(input_op)))
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            output_ops.append(self._output_transform(latent))

        return output_ops
    
def get_signature(dataset, batch_size):
    """
    Get signature of inputs for the training loop.
    The signature is used by tf.function
    """

    input_list = []
    target_list = []
    for _, data in dataset.iterrows():
        dd = make_graph(data, geo_df, 0)
        if dd[0] is not None:
            input_list.append(dd[0])
            target_list.append(dd[1])
            
        if len(input_list) == batch_size:
            break

    inputs = utils_tf.concat(input_list, axis=0)
    targets = utils_tf.concat(target_list, axis=0)
    input_signature = (
      utils_tf.specs_from_graphs_tuple(inputs),
      utils_tf.specs_from_graphs_tuple(targets)
    )
    
    return input_signature

# Our baseline, fully-connected neural network for classification.
# Operates on a single vector (e.g. flattened image from one calorimeter layer).
# Optionally uses dropouts between layers.
def baseline_nn_model(strategy, lr=5e-5, dropout=-1.):
    # create model
    def mod(number_pixels):
        with strategy.scope():    
            model = Sequential()
            used_pixels = number_pixels
            model.add(Dense(number_pixels, input_dim=number_pixels, kernel_initializer='normal', activation='relu'))
            if(dropout > 0.): model.add(Dropout(dropout))
            model.add(Dense(used_pixels, activation='relu'))
            if(dropout > 0.): model.add(Dropout(dropout))
            model.add(Dense(int(used_pixels/2), activation='relu'))
            if(dropout > 0.): model.add(Dropout(dropout))
            model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
            # compile model
            optimizer = Adam(lr=lr)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model
    return mod

# A simple implementation of ResNet.
# As input, this takes multiple images, which may be of different sizes,
# and they are all rescaled to a user-specified size.
def resnet(strategy, channels=6, lr=5e-5, classes=2):
    # create model
    def mod(input_shape):
        with strategy.scope():
            
            # Input images -- one for each channel, each channel's dimensions may be different.
            inputs = [Input((None,None,1),name='input'+str(i)) for i in range(channels)]
            
            # Rescale all the input images, so that their dimensions now match.
            scaled_inputs = [tf.image.resize(x,input_shape,name='scaled_input'+str(i)) for i,x in enumerate(inputs)]
            
            # Now "stack" the images along the channels dimension.
            X = tf.concat(values=scaled_inputs, axis=3, name='concat')
            #print('In:',X.shape)
            
            X = ZeroPadding2D((3,3))(X)
            #print('S0:',X.shape)
            
            # Stage 1
            X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name='bn_conv1')(X)
            X = Activation('relu')(X)
            X = MaxPooling2D((3, 3), strides=(2, 2))(X)
            #print('S1:',X.shape)
            
            # Stage 2
            filters = [64, 64, 256]
            f = 3
            stage = 2
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=1)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            #print('S2:',X.shape)
            
            # Stage 3
            filters = [128, 128, 512]
            f = 3
            stage = 3
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=2)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            X = identity_block(X, f, filters, stage=stage, block='d')
            #print('S3:',X.shape)

            # Stage 4
            filters = [256, 256, 1024]
            f = 3
            stage = 4
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=2)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            X = identity_block(X, f, filters, stage=stage, block='d')
            X = identity_block(X, f, filters, stage=stage, block='e')
            X = identity_block(X, f, filters, stage=stage, block='f')
            #print('S4:',X.shape)

            # Stage 5
            filters = [512, 512, 2048]
            f = 3
            stage = 5
            X = convolutional_block(X, f=f, filters=filters, stage=stage, block='a', s=2)
            X = identity_block(X, f, filters, stage=stage, block='b')
            X = identity_block(X, f, filters, stage=stage, block='c')
            #print('S5:',X.shape)

            # AVGPOOL
            pool_size = (2,2)
            if(X.shape[1] == 1):   pool_size = (1,2)
            elif(X.shape[2] == 1): pool_size = (2,1)
            X = AveragePooling2D(pool_size=pool_size, name="avg_pool")(X)
            #print('S6:',X.shape)

            # output layer
            X = Flatten()(X)
            #print('S7:',X.shape)
            
            X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
            # Create model object.
            model = Model(inputs=inputs, outputs=X, name='ResNet50')
        
            # Compile the model
            optimizer = Adam(lr=lr)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model
    return mod

# A simple combination network -- in practice we may use this to
# "combine" classification scores from single calo-layer networks.
def simple_combine_model(strategy, lr=1e-3):
    # create model
    def mod(n_input=6):
        with strategy.scope():
            model = Sequential()
            model.add(Dense(n_input, input_dim=n_input, kernel_initializer='normal', activation='relu'))
            model.add(Dense(4, activation='relu'))
            model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
            # compile model
            optimizer = Adam(lr=lr)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        return model
    return mod