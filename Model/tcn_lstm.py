import tensorflow as tf
from tensorflow.keras import layers, Model

# Residual Block cho TCN
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2):
        super().__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='causal',
                                   dilation_rate=dilation_rate)
        self.relu1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv1D(filters, kernel_size, padding='causal',
                                   dilation_rate=dilation_rate)
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

# TCN Model
class TCN_Model(tf.keras.Model):
    def __init__(self, num_blocks=4, filters=64, kernel_size=3, target_len=5):
        super().__init__()

        self.tcn_blocks = tf.keras.Sequential()
        for i in range(num_blocks):
            dilation_rate = 2 ** i
            self.tcn_blocks.add(ResidualBlock(filters, kernel_size, dilation_rate))

        # Lấy bước thời gian cuối cùng
        self.last_time_step = layers.Lambda(lambda x: x[:, -1, :])  # shape (batch, features)

        # Fully connected layers
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(target_len)  # output sequence

    def call(self, x, training=False):
        x = self.tcn_blocks(x, training=training)  # (batch, time, filters)
        x = self.last_time_step(x)                 # (batch, filters)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)                         # (batch, target_len)
