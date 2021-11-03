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
            ResBlock(),
            SubPixelConv(double_reduction=self.double_reduction),
            ResBlock()
        ]

    def call(self, inputs, *args, **kwargs):
        z = inputs
        for layer in self.internal:
            z = layer(z)
        return z

class CutBlock (Layer):
    def __init__(self, kernel=3, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.kernel = kernel
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "kernel" : self.kernel}

    def build(self, input_shape):
        self.internal = [
            ConvBNRelu(kernel=self.kernel, stride=1),
            ConvBNRelu(kernel=self.kernel, stride=1),
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
    def __init__(self, depth_memory, nb_resblocks, kernel=3, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.depth_memory = depth_memory
        self.nb_resblocks = nb_resblocks
        self.kernel = kernel
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "kernel" : self.kernel,
            "depth_memory" : self.depth_memory,
            "nb_resblocks" : self.nb_resblocks}

    def build(self, input_shape):
        super(PropagationUnit, self).build(input_shape)

        self.concat_x = Concatenate(axis=-1)

        self.preprocess = [
            ResBlock(depth=self.depth_memory),
            ResBlock(depth=self.depth_memory)
        ]

        build_standard_conv = lambda name : Conv2D(self.depth_memory, kernel_size=7, padding="same", use_bias=True, name=name)
        build_standard_bn = lambda name : BatchNormalization(axis=-1, name=name)
        build_standard_biais = lambda name : self.add_weight(name=name, shape=self.depth_memory, initializer="zeros")
        build_standard_sigm = lambda name : Activation("sigmoid", name=name)
        
        self.conv_xi = build_standard_conv("conv_xi")
        self.conv_ai = build_standard_conv("conv_ai")
        self.conv_mi = build_standard_conv("conv_mi")
        self.bn_xi = build_standard_bn("bn_xi")
        self.bn_ai = build_standard_bn("bn_ai")
        self.bn_mi = build_standard_bn("bn_mi")
        self.biais_i = build_standard_biais("biais_i")
        self.sigm_i = build_standard_sigm("sigm_i")

        self.conv_xf = build_standard_conv("conv_xf")
        self.conv_af = build_standard_conv("conv_af")
        self.conv_mf = build_standard_conv("conv_mf")
        self.bn_xf = build_standard_bn("bn_xf")
        self.bn_af = build_standard_bn("bn_af")
        self.bn_mf = build_standard_bn("bn_mf")
        self.biais_f = build_standard_biais("biais_f")
        self.sigm_f = build_standard_sigm("sigm_f")
        
        self.conv_xm = build_standard_conv("conv_xm")
        self.conv_am = build_standard_conv("conv_am")
        self.bn_xm = build_standard_bn("bn_xm")
        self.bn_am = build_standard_bn("bn_am")
        self.biais_m = build_standard_biais("biais_m")

        self.conv_xo = build_standard_conv("conv_xo")
        self.conv_ao = build_standard_conv("conv_ao")
        self.conv_mo = build_standard_conv("conv_mo")
        self.bn_xo = build_standard_bn("bn_xo")
        self.bn_ao = build_standard_bn("bn_ao")
        self.bn_mo = build_standard_bn("bn_mo")
        self.biais_o = build_standard_biais("biais_o")
        self.sigm_o = build_standard_sigm("sigm_o")
        self.tanh = Activation("tanh", name="tanh")

        self.conv_alpha = Conv2D(1, kernel_size=7, padding="same", use_bias=True, name="conv_alpha")
        self.bn_alpha = build_standard_bn("bn_alpha")
        self.sigm_alpha = build_standard_sigm("sigm_alpha")

        self.concat_end = Concatenate(axis=-1, name="concat_end")


    def call(self, inputs, *args, **kwargs):
        if len(inputs) == 4:
            input_img_and_trimap, trimap, alpha, memory = inputs
        else:
            input_img_and_trimap, trimap, alpha = inputs
            memory = tf.expand_dims(tf.zeros(
                shape = (tf.shape(alpha)[1], tf.shape(alpha)[2], self.depth_memory)
                ), axis=0)

        # Remove user's trimap
        input_img = tf.slice(input_img_and_trimap, [0,0,0,0],[-1, -1, -1, 3])

        # Preprocess
        x = self.concat_x([input_img, trimap, alpha])
        for k in range(self.nb_resblocks):
            x = self.preprocess[k](x)
        
        # Input Gate
        i = self.bn_xi(self.conv_xi(x)) + self.bn_ai(self.conv_ai(alpha)) + self.bn_mi(self.conv_mi(memory)) + self.biais_i
        i = self.sigm_i(i)

        # Forget Gate
        f = self.bn_xf(self.conv_xf(x)) + self.bn_af(self.conv_af(alpha)) + self.bn_mf(self.conv_mf(memory)) + self.biais_f
        f = self.sigm_f(f)

        # Memory update
        memory = f*memory + i*self.tanh(self.bn_xm(self.conv_xm(x)) + self.bn_am(self.conv_am(alpha)) + self.biais_m)

        # Output gate 
        o = self.bn_xo(self.conv_xo(x)) + self.bn_ao(self.conv_ao(alpha)) + self.bn_mo(self.conv_mo(memory)) + self.biais_o
        o = self.sigm_o(o)

        # Output
        alpha = self.sigm_alpha(self.bn_alpha(self.conv_alpha(o*memory)))

        return self.concat_end([alpha, memory])


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1 + self.depth_memory)
    
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

def get_model(depth=32, input_shape=(None, None)):
    observers = []

    ##############
    ### Entree ###
    ##############
    inputs = Input(shape = input_shape + (6,), name="input")
    
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
    l = Add()([l, CutBlock(kernel=3)(deep_cut)])

    l = DecoderBlock(kernel=3, double_reduction=False)(l)
    l = Add()([l, CutBlock(kernel=3)(middle_cut)])

    l = DecoderBlock(kernel=3, double_reduction=False)(l)
    # l = Add()([l, CutBlock()(shallow_cut)])

    l = DecoderBlock(kernel=3, double_reduction=False)(l)

    # Sortie
    l = Conv2D(3, kernel_size=3, padding="same", name="conv_out_trimap")(l)
    # ArgMax = Layer(lambda x : tf.cast(tf.argsort(x, axis=-1, direction="DESCENDING") == 0, dtype="float32"))
    trimap = Softmax(axis=-1)(l)

    m_trimap_only = Model(inputs=inputs, outputs=trimap, name="trimap_decoder")

    #####################
    ### Decoder Alpha ###
    #####################

    l = DecoderBlock(kernel=3, double_reduction=False)(end_encoder)
    # l = Add()([l, CutBlock(kernel=3)(deep_cut)])

    l = DecoderBlock(kernel=3, double_reduction=False)(l)
    l = Add()([l, CutBlock(kernel=3)(middle_cut)])

    l = DecoderBlock(kernel=3, double_reduction=False)(l)
    l = Add()([l, CutBlock()(shallow_cut)])

    l = DecoderBlock(kernel=3, double_reduction=False)(l)

    # Sortie
    alpha = ConvBNRelu(depth=1, kernel=3, name="conv_out_alpha")(l)

    ########################
    ### Propagation Unit ###
    ########################

    depth_memory = depth
    prop = PropagationUnit(nb_resblocks=2, depth_memory = depth_memory)
    memory = None
    # observers.append(Model(inputs, unknown_region, name="mask"))
    observers.append(Model(inputs, alpha, name="alpha_initial"))

    for k in range(3):
        if memory is None:
            alpha_and_memory = prop([inputs, trimap, alpha])
        else:
            alpha_and_memory = prop([inputs, trimap, alpha, memory])
        alpha = Lambda(lambda x : tf.slice(x, [0,0,0,0],[-1, -1, -1, 1]))(alpha_and_memory)
        memory = Lambda(lambda x : tf.slice(x, [0,0,0,1],[-1, -1, -1, depth_memory]))(alpha_and_memory)
        observers.append(Model(inputs, alpha, name=f"refined_alpha_{k+1}"))

    #########################
    ### Add Loss' Weights ###
    #########################
    
    loss_weights = Weights(output_dim=(2,1), initial_value=tf.math.log(4.0).numpy())(inputs)
    m = Model(inputs=inputs, outputs=[trimap, alpha], name="adamatting")
    m_training = Model(inputs=inputs, outputs=[trimap, alpha, loss_weights], name="adamatting_training")
    return m_training, m, m_trimap_only, observers