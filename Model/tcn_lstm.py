import tensorflow as tf
from tensorflow.keras import layers, Model

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

# --- Residual Block ---
@register_keras_serializable(package="Model.tcn_lstm")
class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2):
        super().__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.relu1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.relu2 = layers.Activation('relu')
        self.dropout2 = layers.Dropout(dropout_rate)

        self.downsample = None
        self.final_relu = layers.Activation('relu')
        self.filters = filters

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels != self.filters:
            self.downsample = layers.Conv1D(self.filters, kernel_size=1, padding='same')
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)
        return self.final_relu(x + residual)

# --- TCN + LSTM Model ---
@register_keras_serializable(package="Model.tcn_lstm")
class TCN_LSTM(Model):
    def __init__(self, num_blocks=4 , filters=64, kernel_size=3, lstm_units=64, target_len=5, dropout_rate=0.25):
        super().__init__()

        # TCN stack
        self.tcn_blocks = tf.keras.Sequential([
            ResidualBlock(filters, kernel_size, 2 ** i, dropout_rate)
            for i in range(num_blocks)
        ])

        # LSTM layer
        self.lstm = layers.LSTM(lstm_units, return_sequences=False)

        # Fully connected
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(target_len)

    def call(self, x, training=False):
        x = self.tcn_blocks(x, training=training)
        x = self.lstm(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

