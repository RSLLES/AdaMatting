import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, MaxPooling2D, Add, Concatenate
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras import Model, Sequential

# Il y a 3 divisions : il faut donc que img size soit un multiple de 8
def get_model(img_size, depth=32):

    ##############
    ### Entree ###
    ##############
    inputs = Input(shape = img_size + (6,), name="input")
    
    ########################
    ### Elements de base ###
    ########################

    # Chaque brique retourne :
    # - son model
    # - sa taille de sortie

    def ConvBNRelu_withDepth(shape, depth, kernel=3, stride=1, id=""):
        return Sequential(
            [
                Input(shape = shape),
                Conv2D(depth, kernel, strides=stride, padding="same"),
                BatchNormalization(),
                LeakyReLU()
            ],
            name= f"convbnrelu{id}"
        )

    def ConvBNRelu(shape, kernel=3, stride=1, prefix="", id=""):
        return ConvBNRelu_withDepth(shape=shape, depth=shape[-1], kernel=kernel, stride=stride, id=id)


    def SubPixelConv(shape, id="", double_reduction=False):
        depth = 2*shape[-1] if not double_reduction else shape[-1]
        return Sequential(
            [
                Input(shape = shape),
                Conv2D(depth, 3, strides=1, padding="same"),
                Lambda(
                    lambda tnsr : tf.nn.depth_to_space(tnsr, block_size=2),
                    output_shape = lambda input_shape : (input_shape[0], input_shape[1]*2, input_shape[2]*2, int(depth/4)),
                )
            ],
            name=f"subpixelconv{id}"
        )

    
    def ResBlock(shape, downsample=False, prefix="", id=""):
        downstride = 2 if downsample else 1
        depth = shape[-1]*downstride

        entry = Input(shape = shape)
        
        x = ConvBNRelu_withDepth(shape, depth=depth, stride=downstride, id="1")(entry)
        x = ConvBNRelu(x.shape[1:], id="2")(x)
        
        cut = Conv2D(filters=depth, kernel_size=1, strides=downstride, padding="same", name="cut")(entry)

        x = Add()([x, cut])
        x = LeakyReLU()(x)
        if downsample:
            x = MaxPooling2D(pool_size=3, strides=1, padding="same")(x)

        return  Model(inputs = entry, outputs = x, name = f"resnetdown{id}" if downsample else f"resnet{id}")

    def Concats(id=""):
        return Concatenate(axis=-1, name=f"concat{id}")
    
    
    ###############
    ### Encoder ###
    ###############
    l = inputs

    l = ConvBNRelu_withDepth(shape=l.shape[1:], depth=depth, kernel=3, id="encoder1")(l)
    l = MaxPooling2D(pool_size=3, strides=2, padding="same", name="maxpoolencoder1")(l)

    l = ResBlock(shape=l.shape[1:], id="encoder2")(l)
    shallow_cut = l
    l = ResBlock(shape=l.shape[1:], id="encoder2", downsample=True)(l)

    l = ResBlock(shape=l.shape[1:], id="encoder3")(l)
    middle_cut = l
    l = ResBlock(shape=l.shape[1:], id="encoder3", downsample=True)(l)
    
    l = ResBlock(shape=l.shape[1:], id="encoder4")(l)
    deep_cut = l
    l = ResBlock(shape=l.shape[1:], id="encoder4", downsample=True)(l)


    ###############
    ### Decoder ###
    ###############

    def DecoderBlock(shape, kernel, id="", double_reduction=False):
        inputs = Input(shape = shape)
        l = inputs
        l = ConvBNRelu(shape=l.shape[1:], kernel=kernel)(l)
        l = SubPixelConv(shape=l.shape[1:], double_reduction=double_reduction)(l)

        return Model(inputs=inputs, outputs=l, name=f"decoderblock{id}")

    l = DecoderBlock(shape=l.shape[1:], kernel=7, id="4")(l)
    l = Concats(id="4")([l, deep_cut])

    l = DecoderBlock(shape=l.shape[1:], kernel=7, id="3", double_reduction=True)(l)
    l = Concats(id="3")([l, middle_cut])

    l = DecoderBlock(shape=l.shape[1:], kernel=7, id="2", double_reduction=True)(l)
    l = Concats(id="2")([l, shallow_cut])

    l = DecoderBlock(shape=l.shape[1:], kernel=7, id="1", double_reduction=False)(l)

    # Sortie
    l = Conv2D(3, kernel_size=3, padding="same", name="conv_out")(l)
    adapted_trimap = Softmax(name="softmax_out")(l)

    return Model(inputs=inputs, outputs=adapted_trimap, name="trimap_decoder")