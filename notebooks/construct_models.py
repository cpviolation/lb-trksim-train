import tensorflow as tf

def EfficiencyNN(layers=1,X_shape=None,y_shape=None):
    """basic dense NN for Efficiency model"""
    dense_config = dict(
        units=128,
        activation='tanh', 
        kernel_initializer='he_normal', 
        kernel_regularizer=tf.keras.regularizers.L2(1e-3),
    )
    input = tf.keras.layers.Input(batch_input_shape=[None]+X_shape[1:])
    x = tf.keras.layers.Dense(**dense_config)(input)

    for i in range(layers):
        r = tf.keras.layers.Dense(**dense_config)(x)
        x = tf.keras.layers.Add()([x, r])
    x = tf.keras.layers.Dense(y_shape[1], activation='linear', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Softmax()(x)  ## needed by scikinC

    model = tf.keras.Model(inputs=[input], outputs=[x])
    return model

def AcceptanceNN(layers=1,X_shape=None):
    """basic dense NN for Acceptance model"""
    input = tf.keras.layers.Input(batch_input_shape=[None]+X_shape[1:])
    x = tf.keras.layers.Dense(
        128, activation='tanh',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.L2(1e-3)
    )(input)

    for i in range(layers):
        r = tf.keras.layers.Dense(128, activation='tanh', kernel_initializer='zeros')(x)
        x = tf.keras.layers.Add()([x, r])
    x = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])
    return model

def ResolutionGANgenerator(layers=1,X_shape=None,y_shape=None):
    """basic NN for Resolution generator model"""
    g_input = tf.keras.layers.Input(shape=[X_shape[1]])
    random = tf.keras.layers.Input(shape=[128])

    g_dense_cfg=dict(units=128, activation='tanh', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(1e-3))

    x = tf.keras.layers.Concatenate(axis=1)((g_input, random))
    x = tf.keras.layers.Dense(**g_dense_cfg)(x)
    for i in range(layers):
        x_ = tf.keras.layers.Dense(**g_dense_cfg)(x)
        x = x + x_
        
    ny = y_shape[1]
    output = tf.keras.layers.Dense(ny)(x) #+ random[:, :ny]
    generator = tf.keras.Model(inputs=[g_input, random], outputs = output)
    return generator
        
def ResolutionGANdiscriminator(layers=1,X_shape=None,y_shape=None):
    """basic NN for Resolution discriminator model"""
    d_input_ref_x = tf.keras.layers.Input(shape=[X_shape[1]])

    d_input_ref_y = tf.keras.layers.Input(shape=[y_shape[1]])
    d_input_gen_y = tf.keras.layers.Input(shape=[y_shape[1]])

    d_input_y = tf.keras.layers.Concatenate(axis=0)((d_input_ref_y, d_input_gen_y))
    d_input_x = tf.keras.layers.Concatenate(axis=0)((d_input_ref_x, d_input_ref_x))

    d_input = tf.keras.layers.Concatenate(axis=1)((d_input_x, d_input_y))

    d_dense_cfg=dict(units=128, activation='tanh', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(1e-3))

    x = tf.keras.layers.Dense(**d_dense_cfg)(d_input)
    for i in range(layers):
        x_ = tf.keras.layers.Dense(**d_dense_cfg)(x)
        x = x + x_

    output = tf.keras.layers.Dense(1)(x)
    discriminator = tf.keras.Model(inputs=[d_input_ref_x, d_input_ref_y, d_input_gen_y], outputs=[output])
    return discriminator

def ResolutionGAN(gen_layers=1,discr_layers=1,ref_layers=1,X_shape=None,y_shape=None):
    """define generator and discriminator for Resolution model"""
    generator     = ResolutionGANgenerator(    layers=gen_layers  ,X_shape=X_shape,y_shape=y_shape)
    discriminator = ResolutionGANdiscriminator(layers=discr_layers,X_shape=X_shape,y_shape=y_shape)
    referee       = ResolutionGANdiscriminator(layers=ref_layers  ,X_shape=X_shape,y_shape=y_shape)
    return generator,discriminator,referee


def CovarianceGANgenerator(layers=1,X_shape=None,y_shape=None):
    """basic NN for Resolution generator model"""
    g_input = tf.keras.layers.Input(shape=[X_shape[1]])
    random = tf.keras.layers.Input(shape=[128])

    g_dense_cfg=dict(units=128, activation='tanh', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(1e-3))

    x = tf.keras.layers.Concatenate(axis=1)((g_input, random))
    x = tf.keras.layers.Dense(**g_dense_cfg)(x)
    for i in range(g_nLayers):
        x_ = tf.keras.layers.Dense(**g_dense_cfg)(x)
        x = x + x_

    ny = y_shape[1]
    output = tf.keras.layers.Dense(ny)(x) #+ random[:, :ny]

    generator = tf.keras.Model(inputs=[g_input, random], outputs = output)
    return generator
        
def CovarianceGANdiscriminator(layers=1,X_shape=None,y_shape=None):
    """basic NN for Resolution discriminator model"""
    d_input_ref_x = tf.keras.layers.Input(shape=[X.shape[1]], name="X_ref")

    d_input_ref_y = tf.keras.layers.Input(shape=[y.shape[1]], name="Y_ref")
    d_input_gen_y = tf.keras.layers.Input(shape=[y.shape[1]], name="Y_gen")

    d_input_y = tf.keras.layers.Concatenate(axis=0, name="Y")((d_input_ref_y, d_input_gen_y))
    d_input_x = tf.keras.layers.Concatenate(axis=0, name="X")((d_input_ref_x, d_input_ref_x))

    d_input = tf.keras.layers.Concatenate(axis=1, name="XY")((d_input_x, d_input_y))

    d_dense_cfg=dict(units=128, activation='tanh', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(1e-2))

    x = tf.keras.layers.Dense(**d_dense_cfg)(d_input)
    for i in range(d_nLayers):
        x = tf.keras.layers.Dense(**d_dense_cfg)(x)
        #x = x + x_

    output = tf.keras.layers.Dense(1)(x)

    discriminator = tf.keras.Model(inputs=[d_input_ref_x, d_input_ref_y, d_input_gen_y], outputs=[output])
    return discriminator

def CovarianceGAN(gen_layers=1,discr_layers=1,ref_layers=1,X_shape=None,y_shape=None):
    """define generator and discriminator for Resolution model"""
    generator     = CovarianceGANgenerator(    layers=gen_layers  ,X_shape=X_shape,y_shape=y_shape)
    discriminator = CovarianceGANdiscriminator(layers=discr_layers,X_shape=X_shape,y_shape=y_shape)
    referee       = CovarianceGANdiscriminator(layers=ref_layers  ,X_shape=X_shape,y_shape=y_shape)
    return generator,discriminator,referee