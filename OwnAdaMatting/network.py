from keras.layers.core import Activation
from keras.losses import BinaryCrossentropy
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, MaxPooling2D, Add, Concatenate, Layer

from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras import Model
from keras.initializers import Constant
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
    def __init__(self, depth=None, downsample=False, kernel=3, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.downsample = downsample
        self.stride = 2 if downsample else 1
        self.depth = depth
        self.kernel = kernel
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "downsample" : self.downsample,
            "stride" : self.stride,
            "depth" : self.depth,
            "kernel" : self.kernel}

    def build(self, input_shape):
        if self.depth == None:
            self.depth = input_shape[-1]*2 if self.downsample else self.depth
        else:
            self.depth = self.depth*2 if self.downsample else self.depth
        
        self.conv1 = ConvBNRelu(depth=self.depth, kernel=self.kernel, stride=self.stride)
        self.conv2 = ConvBNRelu(depth=None, kernel=self.kernel, stride=1)
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

class PropagationUnit (Layer):
    def __init__(self, depth=None, kernel=3, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.depth = depth
        self.kernel = kernel
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "depth" : self.depth,
            "kernel" : self.kernel}

    def build(self, input_shape):
        _, _, alpha_estimation_shape = input_shape
        self.depth = alpha_estimation_shape[-1] if self.depth == None else self.depth

        self.preprocess_i = ResBlock()
        self.concatenate_img_trimap_alpha = Concatenate(axis=-1)
        self.concatenate_hi = Concatenate(axis=-1)
        self.concatenate_ui = Concatenate(axis=-1)
        self.sigm = Activation("sigmoid")
        self.activation = Activation("tanh")
        self.conv_z = Conv2D(self.depth, kernel_size=self.kernel, padding="same")
        self.conv_r = Conv2D(self.depth, kernel_size=self.kernel, padding="same")
        self.conv_hh = Conv2D(self.depth, kernel_size=self.kernel, padding="same")

    def call(self, inputs, *args, **kwargs):
        input_img_and_trimap,  adapted_trimap, current_alpha_estimation = inputs

        # Remove input user's trimap
        input_img = tf.slice(input_img_and_trimap, [0,0,0,0],[-1, -1, -1, 3])

        # Preprocess
        i = self.concatenate_img_trimap_alpha([input_img, adapted_trimap, current_alpha_estimation])
        i = self.preprocess_i(i)

        # Concatenate h and i
        h = current_alpha_estimation
        hi = self.concatenate_hi([h, i])

        # Update Gate
        z = self.sigm(self.conv_z(hi))

        # Reset Gate
        r = self.sigm(self.conv_r(hi))

        # Candidate Alpha
        u = r*h
        ui = self.concatenate_ui([u, i])
        hh = self.activation(self.conv_hh(ui))

        # Merging
        return (1-z)*h + z*hh


    def compute_output_shape(self, input_shape):
        _, _, alpha_estimation_shape = input_shape
        return alpha_estimation_shape
    
#############################
### Custom Multitask Loss ###
#############################

class Weights(Layer):
    def __init__(self, output_dim, initial_value=1.0, **kwargs):
       self.output_dim = output_dim
       self.initial_value = initial_value
       super(Weights, self).__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "kernel" : self.kernel,
            "output_dim" : self.output_dim,
            "initial_value" : self.initial_value}

    def build(self, input_shapes):
       self.kernel = self.add_weight(
           name='kernel', 
           shape=self.output_dim, 
           initializer=Constant(self.initial_value), 
           trainable=True)
       super(Weights, self).build(input_shapes)  

    def call(self, inputs=None):
       return self.kernel

    def compute_output_shape(self):
       return self.output_dim


class MultiTaskLoss(Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="multitaskloss"):
        self.bce = BinaryCrossentropy(from_logits=False)
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None, eps=1e-6):
        gt_trimap = tf.slice(y_true, [0,0,0,0], [-1, -1, -1, 3])
        gt_alpha =  tf.slice(y_true, [0,0,0,3], [-1, -1, -1, 1])
        trimap, alpha, weights = y_pred

        log_s1_sqr = tf.squeeze(tf.slice(weights, [0,0], [1,1]))
        log_s2_sqr = tf.squeeze(tf.slice(weights, [1,0], [1,1]))
        s1_sqr = tf.exp(log_s1_sqr)
        s2_sqr = tf.exp(log_s2_sqr)

        # Loss de l'alpha (ponderes par les pixels gris estimes)
        grey = tf.slice(trimap, [0,0,0,1], [-1, -1, -1, 1])
        loss_alpha = tf.reduce_sum(tf.abs(alpha - gt_alpha)*grey)/(tf.reduce_sum(grey) + eps) + eps

        # Loss de la trimap : Binary cross entropy
        loss_trimap = self.bce(y_true=gt_trimap, y_pred=trimap)

        # Fusion
        return loss_trimap/(s1_sqr + eps) + loss_alpha/(s2_sqr + eps) + log_s1_sqr + log_s2_sqr
        

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
    l = Conv2D(1, kernel_size=3, padding="same", name="conv_out_alpha")(l)
    end_alpha_decoder = Activation("sigmoid")(l)

    ########################
    ### Propagation Unit ###
    ########################
    prop = PropagationUnit(depth=1, kernel=7)
    alpha = end_alpha_decoder
    for _ in range(3):
        alpha = prop([inputs, end_trimap_decoder, alpha])
    out = alpha

    #########################
    ### Add Loss' Weights ###
    #########################
    loss_weights = Weights(output_dim=(2,1), initial_value=2.0*tf.math.log(4.0))(inputs)

    return Model(inputs=inputs, outputs=[end_trimap_decoder, out, loss_weights], name="trimap_decoder")