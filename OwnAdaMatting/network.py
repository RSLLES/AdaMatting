import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, MaxPooling2D, Add, Concatenate, Layer
from keras.layers.convolutional_recurrent import ConvLSTMCell

from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras import Model, Sequential

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

class PropagationUnit (Layer):
    def __init__(self, depth=None, kernel=3, stride=1, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config
            }

    def build(self, input_shape):
        self.rs = ResBlock(downsample=False)
        self.lstmconvcell = ConvLSTMCell(rank=2, filters=3, kernel_size=3, padding="same")

        self.built = True

    def call(self, inputs, states, *args, **kwargs):
        z = inputs
        for layer in self.internal:
            z = layer(z)
        return z



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

    l = ResBlock(downsample=False)(l)
    shallow_cut = l
    l = ResBlock(downsample=True)(l)

    l = ResBlock(downsample=False)(l)
    middle_cut = l
    l = ResBlock(downsample=True)(l)
    
    l = ResBlock(downsample=False)(l)
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
    l = Conv2D(3, kernel_size=3, padding="same", name="conv_out")(l)
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
    l = Conv2D(3, kernel_size=3, padding="same", name="conv_out")(l)
    end_alpha_decoder = Softmax()(l)


    ########################
    ### Propagation Unit ###
    ########################


    return Model(inputs=inputs, outputs=end_trimap_decoder, name="trimap_decoder")