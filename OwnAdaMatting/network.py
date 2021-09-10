import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

# Il y a 3 divisions : il faut donc que img size soit un multiple de 8
def get_model(img_size):
    inputs = [layers.Input(shape=img_size + (3,)), layers.Input(shape=img_size + (3,))]
    depth = 32

    # Elements de base
    def ConvBNRelu(entry, dim, kernel=3, stride=1):
        x = layers.Conv2D(dim, kernel, strides=stride, padding="same")(entry)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def SubPixelConv(entry, dim):
        x = layers.Conv2D(dim*2, 3, strides=1, padding="same")(entry)
        x = layers.Lambda(  lambda tnsr : tf.nn.depth_to_space(tnsr, block_size=2),
                            output_shape=lambda input_shape : (input_shape[0], input_shape[1]*2, input_shape[2]*2, dim/2))(x)
        return x
    
    def res_block(entry, dim, downsample=False):
        # 2 blocks normaux
        x = entry
        downstride = 2 if downsample else 1
        # Premier block descend si necessaire
        x = ConvBNRelu(x, dim, stride = downstride)
        # Deuxieme normal
        x = ConvBNRelu(x, dim)
        
        # De l'autre, un cut quasi direct
        cut = layers.Conv2D(filters=dim, kernel_size=1, strides=downstride, padding="same")(entry)

        # Somme et retour
        x = layers.Add()([x, cut])
        x = layers.Activation("relu")(x)
        if downsample:
            x = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")(x)
        return x
    
    # Encoder
    l = layers.Concatenate(axis=-1)(inputs)
    l = ConvBNRelu(l, depth)
    l = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")(l)

    l = res_block(l, depth)
    shallow_cut = l
    l = res_block(l, depth, downsample=True)

    l = res_block(l, depth*2)
    middle_cut = l
    l = res_block(l, depth*2, downsample=True)
    
    l = res_block(l, depth*4)
    deep_cut = l
    l = res_block(l, depth*4, downsample=True)

    # Decoder
    l = ConvBNRelu(l, depth*8, kernel=7)
    l = SubPixelConv(l, depth*8)

    l = layers.Concatenate(axis=-1)([l, deep_cut])
    l = ConvBNRelu(l, depth*4, kernel=7)
    l = SubPixelConv(l, depth*4)
    
    l = layers.Concatenate(axis=-1)([l, middle_cut])
    l = ConvBNRelu(l, depth*2, kernel=7)
    l = SubPixelConv(l, depth*2)

    l = layers.Concatenate(axis=-1)([l, shallow_cut])
    l = ConvBNRelu(l, depth, kernel=7)

    # Sortie
    l = layers.Conv2D(3, kernel_size=3, padding="same")(l)
    adapted_trimap = layers.Activation("softmax")(l)

    return Model(inputs=inputs, outputs=adapted_trimap)