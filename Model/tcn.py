import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Version-compatible decorator
try:
    # TensorFlow 2.13+
    register_keras_serializable = tf.keras.saving.register_keras_serializable
except AttributeError:
    try:
        # TensorFlow 2.0-2.12
        register_keras_serializable = tf.keras.utils.register_keras_serializable
    except AttributeError:
        # Fallback: no-op decorator for older versions
        def register_keras_serializable(package=None, name=None):
            def decorator(cls):
                return cls
            return decorator


@register_keras_serializable(package="Model.tcn")
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, l2_rate=0.001, **kwargs):

        super().__init__(**kwargs) 


        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.l2_rate = l2_rate
    

        l2_reg = regularizers.l2(self.l2_rate) if self.l2_rate > 0 else None

        self.conv1 = layers.Conv1D(filters, kernel_size, padding='causal',
                                  dilation_rate=dilation_rate,
                                  kernel_regularizer=l2_reg)
        self.relu1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv1D(filters, kernel_size, padding='causal',
                                  dilation_rate=dilation_rate,
                                  kernel_regularizer=l2_reg)
        self.relu2 = layers.Activation('relu')
        self.dropout2 = layers.Dropout(dropout_rate)

        self.downsample = None
        self.final_relu = layers.Activation('relu')

    def build(self, input_shape):
        in_channels = input_shape[-1]

        if in_channels != self.filters:
            l2_reg = regularizers.l2(self.l2_rate) if self.l2_rate > 0 else None
            self.downsample = layers.Conv1D(self.filters, kernel_size=1, padding='same', 
                                            kernel_regularizer=l2_reg)
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)

        return self.final_relu(x + residual)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate,
            'l2_rate': self.l2_rate
        })
        return config


@register_keras_serializable(package="Model.tcn")
class TCN_Model(tf.keras.Model):
    def __init__(self, num_blocks=4, filters=64, kernel_size=3, target_len=5, 
                 dropout_rate=0.15, l2_rate=0.003, **kwargs):
        super().__init__(**kwargs)

        # Store parameters for get_config
        self.num_blocks = num_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.target_len = target_len
        self.dropout_rate = dropout_rate
        self.l2_rate = l2_rate

        self.tcn_blocks = tf.keras.Sequential()
        for i in range(num_blocks):
            dilation_rate = 2 ** i
            self.tcn_blocks.add(ResidualBlock(
                filters, kernel_size, dilation_rate, 
                dropout_rate=self.dropout_rate, 
                l2_rate=self.l2_rate
            ))

        self.last_time_step = layers.Lambda(lambda x: x[:, -1, :])  # shape (batch, features)

        l2_reg = regularizers.l2(self.l2_rate) if self.l2_rate > 0 else None

        # Fully connected layers
        self.fc1 = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)
        self.fc2 = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg)
        self.out = layers.Dense(target_len)  # output sequence

    def call(self, x, training=False):
        x = self.tcn_blocks(x, training=training)  # (batch, time, filters)
        x = self.last_time_step(x)                # (batch, filters)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)                        # (batch, target_len)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_blocks': self.num_blocks,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'target_len': self.target_len,
            'dropout_rate': self.dropout_rate,
            'l2_rate': self.l2_rate
        })
        return config



__all__ = ['ResidualBlock', 'TCN_Model']