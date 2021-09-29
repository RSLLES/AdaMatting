from keras.layers.core import Activation
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, MaxPooling2D, Add, Concatenate, Layer

from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras import Model
from keras.initializers import Constant

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
    def __init__(self, depth_alpha, depth_memory, kernel=3, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.depth_alpha = depth_alpha
        self.depth_memory = depth_memory
        self.kernel = kernel
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "kernel" : self.kernel,
            "depth_alpha" : self.depth_alpha,
            "depth_memory" : self.depth_memory}

    def build(self, input_shape):
        self.concatenate_img_trimap_alpha = Concatenate(axis=-1)
        self.concatenate_hi = Concatenate(axis=-1)
        self.concatenate_end = Concatenate(axis=-1)

        self.preprocess_i = ResBlock(depth=self.depth_memory, kernel=7)
        self.conv_gate = Conv2D(3*self.depth_memory, kernel_size=self.kernel, padding="same")
        self.conv_output = Conv2D(self.depth_alpha, kernel_size=self.kernel, padding="same")

        self.sigm = Activation("sigmoid")
        self.activation = Activation("tanh")


    def call(self, inputs, *args, **kwargs):
        input_img_and_trimap, trimap, alpha_and_memory, mask = inputs

        # Remove user's trimap
        input_img = tf.slice(input_img_and_trimap, [0,0,0,0],[-1, -1, -1, 3])
        alpha = tf.slice(alpha_and_memory, [0,0,0,0],[-1, -1, -1, self.depth_alpha])
        memory = tf.slice(alpha_and_memory, [0,0,0,self.depth_alpha],[-1, -1, -1, self.depth_memory])

        # Preprocess
        i = self.concatenate_img_trimap_alpha([input_img, trimap, alpha])
        new_info = self.preprocess_i(i)

        # Concatenate h and i
        h = alpha
        hi = self.concatenate_hi([h, i])

        # Convolution and split into gates
        res = self.conv_gate(hi)
        forget_gate = self.sigm(tf.slice(res, [0,0,0,0*self.depth_memory], [-1, -1, -1, self.depth_memory]))
        update_gate = self.sigm(tf.slice(res, [0,0,0,1*self.depth_memory], [-1, -1, -1, self.depth_memory]))
        output_gate = self.sigm(tf.slice(res, [0,0,0,2*self.depth_memory], [-1, -1, -1, self.depth_memory]))

        # New memory
        new_memory = forget_gate*memory + update_gate*new_info
        
        # Building new alpha
        update = self.activation(self.conv_output(new_info*output_gate))
        # new_alpha = alpha + mask*update
        new_alpha = alpha + update

        # Merging
        return self.concatenate_end([new_alpha, new_memory])


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.depth_alpha + self.depth_memory)
    
class TrimapToTrivialAlpha (Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        return super().get_config()

    def call(self, inputs, *args, **kwargs):
        trimap = inputs
        # bg = tf.slice(trimap, [0,0,0,0],[-1, -1, -1, 1]) # Useless because 0.0 is the default value, see right below
        uk = tf.slice(trimap, [0,0,0,1],[-1, -1, -1, 1])
        fg = tf.slice(trimap, [0,0,0,2],[-1, -1, -1, 1])

        return fg + 0.5*uk  # + 0.0*bg

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)

class GetUnknownRegionsMap (Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        return super().get_config()

    def call(self, inputs, *args, **kwargs):
        trivial_alpha = inputs
        return tf.cast(tf.abs(trivial_alpha - 0.5) < 0.1, dtype="float32") #Pour les erreurs d'arrondies

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)
        
class ArgMax (Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        return super().get_config()

    def build( self, input_shape ):
        return super(ArgMax, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        trimap_sorted = tf.argsort(inputs, axis=-1, direction="DESCENDING")
        return tf.cast(trimap_sorted == 0, dtype="float32")

    def compute_output_shape(self, input_shape):
        return input_shape

class Weights(Layer):
    def __init__(self, output_dim, initial_value=1.0, trainable = True, **kwargs):
       self.output_dim = output_dim
       self.initial_value = initial_value
       self.trainable = trainable
       super(Weights, self).__init__(**kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "kernel" : self.kernel,
            "output_dim" : self.output_dim,
            "initial_value" : self.initial_value,
            "trainable" : self.trainable}

    def build(self, input_shapes):
       self.kernel = self.add_weight(
           name='kernel', 
           shape=self.output_dim, 
           initializer=Constant(self.initial_value), 
           trainable=self.trainable)
       super(Weights, self).build(input_shapes)  

    def call(self, inputs=None):
       return self.kernel

    def compute_output_shape(self):
       return self.output_dim


########################
### MODEL GENERATION ###
########################

# Il y a 3 divisions : il faut donc que img size soit un multiple de 8
def get_model(img_size, depth=32):

    observers = []

    ##############
    ### Entree ###
    ##############
    inputs = Input(shape = (None, None, 6), name="input")
    
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
    # ArgMax = Layer(lambda x : tf.cast(tf.argsort(x, axis=-1, direction="DESCENDING") == 0, dtype="float32"))
    trimap = Softmax(axis=-1)(l)

    #####################
    ### Decoder Alpha ###
    #####################

    l = DecoderBlock(kernel=3, double_reduction=False)(end_encoder)
    # l = Concatenate()([l, deep_cut])

    l = DecoderBlock(kernel=3, double_reduction=False)(l)
    l = Concatenate()([l, middle_cut])

    l = DecoderBlock(kernel=3, double_reduction=True)(l)
    l = Concatenate()([l, shallow_cut])

    l = DecoderBlock(kernel=3, double_reduction=True)(l)

    # Sortie
    end_alpha_decoder = ConvBNRelu(depth=3, kernel=3, name="conv_out_alpha")(l)

    ########################
    ### Propagation Unit ###
    ########################

    prop = PropagationUnit(depth_alpha = 1, depth_memory = 3)
    alpha = TrimapToTrivialAlpha()(trimap)
    memory = end_alpha_decoder
    unknown_region = GetUnknownRegionsMap()(alpha)
    observers.append(Model(inputs, unknown_region, name="mask"))
    observers.append(Model(inputs, alpha, name="alpha_trivial"))

    alpha_and_memory = Concatenate(axis=-1)([alpha, memory])
    for k in range(2):
        alpha_and_memory = prop([inputs, trimap, alpha_and_memory, unknown_region])
        alpha = Lambda(lambda x : tf.slice(x, [0,0,0,0],[-1, -1, -1, 1]))(alpha_and_memory)
        observers.append(Model(inputs, alpha, name=f"refined_alpha_{k+1}"))

    #########################
    ### Add Loss' Weights ###
    #########################
    
    loss_weights = Weights(output_dim=(2,1), initial_value=tf.math.log(4.0).numpy())(inputs)
    return Model(inputs=inputs, outputs=[trimap, alpha, loss_weights], name="trimap_decoder"), observers