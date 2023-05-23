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