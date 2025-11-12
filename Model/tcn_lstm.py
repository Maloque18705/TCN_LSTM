import tensorflow as tf
from tensorflow.keras import layers, Model

# --- Residual Block ---
class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, **kwargs):
        # 1. Thêm **kwargs và truyền vào super()
        super().__init__(**kwargs)
        
        # 2. Lưu các tham số để dùng trong get_config
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        self.conv1 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.relu1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.relu2 = layers.Activation('relu')
        self.dropout2 = layers.Dropout(dropout_rate)

        self.downsample = None
        self.final_relu = layers.Activation('relu')

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

    # 3. Thêm get_config
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout_rate": self.dropout_rate,
        })
        return config

# --- TCN + LSTM Model ---
class TCN_LSTM(Model):
    def __init__(self, num_blocks=4 , filters=64, kernel_size=3, lstm_units=64, target_len=5, dropout_rate=0.15, **kwargs):
        # 1. Thêm **kwargs và truyền vào super()
        super().__init__(**kwargs)
        
        # 2. Lưu các tham số để dùng trong get_config
        self.num_blocks = num_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.target_len = target_len
        self.dropout_rate = dropout_rate

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

    # 3. Thêm get_config
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_blocks": self.num_blocks,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "lstm_units": self.lstm_units,
            "target_len": self.target_len,
            "dropout_rate": self.dropout_rate,
        })
        return config