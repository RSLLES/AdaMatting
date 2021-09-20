from keras.losses import BinaryCrossentropy
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, MaxPooling2D, Add, Concatenate, Layer
from keras.layers.convolutional_recurrent import ConvLSTM, ConvLSTMCell, ConvRNN

from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras import Model
from keras import regularizers
from keras.backend import expand_dims
from keras.losses import Loss

#####################
### CUSTOM LAYERS ###
#####################

class ConvBNRelu (Layer):
    def __init__(self, depth=None, kernel=3, stride=1, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.depth = depth
        self.kernel = kernel
        self.stride = stride
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "depth" : self.depth,
            "kernel" : self.kernel,
            "stride" : self.stride}

    def build(self, input_shape):
        self.depth = input_shape[-1] if self.depth == None else self.depth
        self.internal = [
            Conv2D(self.depth, self.kernel, strides=self.stride, padding="same"),
            BatchNormalization(),
            LeakyReLU()
        ]

    def call(self, inputs, *args, **kwargs):
        z = inputs
        for layer in self.internal:
            z = layer(z)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.depth)

class ResBlock (Layer):
    def __init__(self, downsample=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.downsample = downsample
        self.stride = 2 if downsample else 1
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "downsample" : self.downsample,
            "stride" : self.stride}

    def build(self, input_shape):
        self.depth = input_shape[-1]*2 if self.downsample else None
        
        self.conv1 = ConvBNRelu(depth=self.depth, kernel=3, stride=self.stride)
        self.conv2 = ConvBNRelu(depth=None, kernel=3, stride=1)
        self.convcut = ConvBNRelu(depth=self.depth, kernel=1, stride=self.stride)
        self.add = Add()
        self.relu = LeakyReLU()
        if self.downsample:
            self.maxpool = MaxPooling2D(pool_size=3, strides=1, padding="same")
        
    def call(self, inputs, *args, **kwargs):
        z1 = self.conv1(self.conv2(inputs))
        z2 = self.convcut(inputs)
        z = self.add([z1, z2])
        if self.downsample:
            z = self.maxpool(z)
        return z

class DecoderBlock (Layer):
    def __init__(self, kernel=3, double_reduction=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.double_reduction = double_reduction
        self.kernel = kernel
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "double_reduction" : self.double_reduction,
            "kernel" : self.kernel}

    def build(self, input_shape):
        self.internal = [
            ConvBNRelu(kernel=self.kernel, stride=1),
            SubPixelConv(double_reduction=self.double_reduction)
        ]

    def call(self, inputs, *args, **kwargs):
        z = inputs
        for layer in self.internal:
            z = layer(z)
        return z

class SubPixelConv (Layer):
    def __init__(self, double_reduction=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.double_reduction = double_reduction
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "double_reduction" : self.double_reduction}

    def build(self, input_shape):
        self.depth = 2*input_shape[-1] if not self.double_reduction else input_shape[-1]
        self.internal = [
            Conv2D(self.depth, kernel_size=1, strides=1, padding="same"),
            Lambda(
                lambda tnsr : tf.nn.depth_to_space(tnsr, block_size=2),
                output_shape = lambda input_shape : self.compute_output_shape(input_shape),
            )
        ]

    def call(self, inputs, *args, **kwargs):
        z = inputs
        for layer in self.internal:
            z = layer(z)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*2, input_shape[2]*2, int(self.depth/4))

class PropagationCell (ConvLSTMCell):
    def build(self, input_shape):
        self.rs = ResBlock(downsample=False)
        self.conv = ConvBNRelu(depth=12) # 12 = 3 (img) + 3 (trimap donnee) + 3 (trimap refined) + 3 (decoder alpha)
        super().build(input_shape)

    def call(self, inputs, states, *args, **kwargs):
        last_alpha = states[0]
        l = Concatenate(axis=-1)([inputs, last_alpha])
        l = self.rs(l)
        l = self.conv(l)
        return super().call(l, states, *args, **kwargs)

class Propagation (ConvLSTM):
    def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               dropout=0.0,
               recurrent_dropout=0.0,
               **kwargs):
        cell = PropagationCell(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'))
        super(ConvLSTM, self).__init__(
            2,
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

#############################
### Custom Multitask Loss ###
#############################

class Weights(Layer):
    def __init__(self, output_dim, **kwargs):
       self.output_dim = output_dim
       super(Weights, self).__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "kernel" : self.kernel,
            "output_dim" : self.output_dim}

    def build(self, input_shapes):
       self.kernel = self.add_weight(name='kernel', shape=self.output_dim, initializer='uniform', trainable=True)
       super(Weights, self).build(input_shapes)  

    def call(self, inputs=None):
       return self.kernel

    def compute_output_shape(self):
       return self.output_dim


class MultiTaskLoss(Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="multitaskloss"):
        self.bce = BinaryCrossentropy(from_logits=False)
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        gt_trimap = tf.slice(y_true, [0,0,0,0], [-1, -1, -1, 3])
        gt_alpha =  tf.slice(y_true, [0,0,0,3], [-1, -1, -1, 1])
        trimap, alpha, weights = y_pred

        log_s1 = tf.squeeze(tf.slice(weights, [0,0], [1,1]))
        log_s2 = tf.squeeze(tf.slice(weights, [1,0], [1,1]))
        s1_2 = tf.square(tf.exp(log_s1))
        s2 = tf.exp(log_s2)

        # Loss de l'alpha (ponderes par les pixels gris estimes)
        grey = tf.slice(trimap, [0,0,0,1], [-1, -1, -1, 1])
        loss_alpha = tf.reduce_sum(tf.abs(alpha - gt_alpha)*grey)/tf.reduce_sum(grey)

        # Loss de la trimap : Binary cross entropy
        loss_trimap = self.bce(y_true=gt_trimap, y_pred=trimap)

        # Fusion
        return 0.5*loss_trimap/s1_2 + loss_alpha/s2 + log_s1 + log_s2
        

########################
### MODEL GENERATION ###
########################

# Il y a 3 divisions : il faut donc que img size soit un multiple de 8
def get_model(img_size, depth=32):

    ##############
    ### Entree ###
    ##############
    inputs = Input(shape = img_size + (6,), name="input")
    
    ###############
    ### Encoder ###
    ###############
    l = inputs

    l = ConvBNRelu(depth=depth, kernel=3, stride=1, name="")(l)
    l = ConvBNRelu(depth=depth*2, kernel=3, stride=1, name="")(l)
    l = MaxPooling2D(pool_size=3, strides=2, padding="same", name="")(l)

    # l = ResBlock(downsample=False)(l)
    shallow_cut = l
    l = ResBlock(downsample=True)(l)

    # l = ResBlock(downsample=False)(l)
    middle_cut = l
    l = ResBlock(downsample=True)(l)
    
    # l = ResBlock(downsample=False)(l)
    deep_cut = l
    end_encoder = ResBlock(downsample=True)(l)


    ######################
    ### Decoder Trimap ###
    ######################

    l = DecoderBlock(kernel=3, double_reduction=False)(end_encoder)
    l = Concatenate()([l, deep_cut])

    l = DecoderBlock(kernel=3, double_reduction=True)(l)
    l = Concatenate()([l, middle_cut])

    l = DecoderBlock(kernel=3, double_reduction=True)(l)
    # l = Concatenate()([l, shallow_cut])

    l = DecoderBlock(kernel=3, double_reduction=False)(l)

    # Sortie
    l = Conv2D(3, kernel_size=3, padding="same", name="conv_out_trimap")(l)
    end_trimap_decoder = Softmax()(l)


    #####################
    ### Decoder Alpha ###
    #####################

    l = DecoderBlock(kernel=3, double_reduction=False)(end_encoder)
    # l = Concatenate()([l, deep_cut])

    l = DecoderBlock(kernel=3, double_reduction=True)(l)
    l = Concatenate()([l, middle_cut])

    l = DecoderBlock(kernel=3, double_reduction=True)(l)
    l = Concatenate()([l, shallow_cut])

    l = DecoderBlock(kernel=3, double_reduction=False)(l)

    # Sortie
    l = Conv2D(3, kernel_size=3, padding="same", name="conv_out_alpha")(l)
    end_alpha_decoder = Softmax()(l)


    ########################
    ### Propagation Unit ###
    ########################
    all = Concatenate(axis=-1)([inputs, end_alpha_decoder, end_trimap_decoder])
    all_with_t = expand_dims(all, axis=1)
    all_through_time = Concatenate(axis=1)([all_with_t, all_with_t, all_with_t])
    out = Propagation(filters=1, kernel_size=3, padding="same")(all_through_time)

    #########################
    ### Add Loss' Weights ###
    #########################
    loss_weights = Weights((2,1))(inputs)

    return Model(inputs=inputs, outputs=[end_trimap_decoder, out, loss_weights], name="trimap_decoder")