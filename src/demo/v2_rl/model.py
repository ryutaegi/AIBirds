import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, initializers


def _get_noise_vector(size):
    """Factorized Gaussian noise: f(x) = sign(x) * sqrt(|x|)"""
    x = tf.random.normal((size,))
    return tf.sign(x) * tf.sqrt(tf.abs(x))


class NoisyDense(layers.Layer):
    """
    Factorized Noisy Dense layer (Rainbow 논문 방식, 원본 AIBirds와 동일).
    noise_active=True이면 항상 노이즈 추가 (학습 + 행동 선택 시).
    평가 시에만 set_noisy(False)로 끔.
    """
    def __init__(self, units, noise_std_init=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.noise_std_init = noise_std_init
        self.noise_active = True
        self._activation = activation

    def build(self, input_shape):
        size_in = int(input_shape[-1])
        norm = 1.0 / np.sqrt(size_in)

        uniform_init = initializers.RandomUniform(-norm, norm)
        sigma_init_val = self.noise_std_init * norm

        self.mu = self.add_weight("mu", shape=(self.units, size_in),
                                   initializer=uniform_init, trainable=True)
        self.mu_bias = self.add_weight("mu_bias", shape=(self.units,),
                                        initializer=initializers.Constant(sigma_init_val),
                                        trainable=True)
        self.sigma = self.add_weight("sigma", shape=(self.units, size_in),
                                      initializer=uniform_init, trainable=True)
        self.sigma_bias = self.add_weight("sigma_bias", shape=(self.units,),
                                           initializer=initializers.Constant(sigma_init_val),
                                           trainable=True)
        self.epsilon = tf.Variable(tf.zeros((self.units, size_in)),
                                    trainable=False, name="epsilon")
        self.epsilon_bias = tf.Variable(tf.zeros(self.units),
                                         trainable=False, name="epsilon_bias")
        self._size_in = size_in
        self.reset_noise()
        super().build(input_shape)

    def reset_noise(self):
        eps_in = _get_noise_vector(self._size_in)
        eps_out = _get_noise_vector(self.units)
        self.epsilon.assign(tf.tensordot(eps_out, eps_in, axes=0))
        self.epsilon_bias.assign(eps_out)

    def set_noisy(self, active: bool):
        self.noise_active = active

    def call(self, x, training=None, mask=None):
        if self.noise_active:
            A = self.mu + self.sigma * self.epsilon
            b = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            A = self.mu
            b = self.mu_bias
        out = tf.linalg.matvec(A, x) + b
        if self._activation is not None:
            out = tf.keras.activations.get(self._activation)(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "noise_std_init": self.noise_std_init,
                        "activation": self._activation})
        return config


class SpatialAttention(layers.Layer):
    """
    Spatial Attention (CBAM 논문 기반).
    CNN 특징맵의 어디가 중요한지 학습 -> 돼지/구조물 위치에 자동 집중.
    처음 보는 맵에서도 중요한 영역을 찾아내는 일반화 능력 향상.
    """
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            1, (self.kernel_size, self.kernel_size),
            padding='same', activation='sigmoid',
            kernel_initializer='glorot_normal',
        )
        super().build(input_shape)

    def call(self, x, training=False):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention_map = self.conv(concat)
        return x * attention_map

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class ClassicConvStem(layers.Layer):
    """
    CNN Stem with Spatial Attention + Dropout (일반화 향상 버전).
    원본 AIBirds ClassicConv 기반 + 처음 보는 맵 대응력 강화.

    개선사항:
    - SpatialAttention: 마지막 Conv 후 적용, 돼지/구조물 위치 자동 집중
    - Dropout: Dense 후 적용, 학습 레벨 과적합 방지
    """
    def __init__(self, latent_dim=256, dropout_rate=0.15, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Conv blocks (원본과 동일)
        self.conv1 = layers.Conv2D(32, (4, 4), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_normal')
        self.pool1 = layers.MaxPool2D((2, 2))

        self.conv2 = layers.Conv2D(64, (3, 3), strides=2, padding='same', use_bias=False, kernel_initializer='glorot_normal')
        self.pool2 = layers.MaxPool2D((2, 2))

        self.conv3 = layers.Conv2D(64, (2, 2), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_normal')
        self.pool3 = layers.MaxPool2D((2, 2))

        self.conv4 = layers.Conv2D(128, (2, 2), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_normal')
        self.pool4 = layers.MaxPool2D((2, 2))

        # Spatial Attention: 마지막 Conv 특징맵에서 중요 영역 강조
        self.spatial_attn = SpatialAttention()

        self.flatten = layers.Flatten()
        self.dense_latent = layers.Dense(latent_dim, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.concat = layers.Concatenate()

    def call(self, inputs, training=False):
        x, bird_info = inputs

        x = tf.nn.relu(self.conv1(x))
        x = self.pool1(x)

        x = tf.nn.relu(self.conv2(x))
        x = self.pool2(x)

        x = tf.nn.relu(self.conv3(x))
        x = self.pool3(x)

        x = tf.nn.relu(self.conv4(x))
        x = self.pool4(x)

        # Spatial Attention: "어디를 봐야 하는가" 학습
        x = self.spatial_attn(x, training=training)

        x = self.flatten(x)
        x = self.dense_latent(x)
        x = self.dropout(x, training=training)

        return self.concat([x, bird_info])

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim, "dropout_rate": self.dropout_rate})
        return config


class DuelingQHead(layers.Layer):
    """
    Dueling architecture + Noisy Nets (원본 DoubleQNetwork과 동일 구조).
    stem -> v_hidden(256, noisy) -> v(1, noisy)
    stem -> a_hidden(256, noisy) -> a(num_actions, noisy)
    """
    def __init__(self, num_actions, latent_v_dim=256, latent_a_dim=256,
                 noise_std_init=0.5, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        self.latent_v_dim = latent_v_dim
        self.latent_a_dim = latent_a_dim
        self.noise_std_init = noise_std_init
        self.activation = activation

        if noise_std_init > 0:
            self.v_h = NoisyDense(latent_v_dim, noise_std_init, activation=activation)
            self.v   = NoisyDense(1, noise_std_init)
            self.a_h = NoisyDense(latent_a_dim, noise_std_init, activation=activation)
            self.a   = NoisyDense(num_actions, noise_std_init)
        else:
            self.v_h = layers.Dense(latent_v_dim, activation=activation)
            self.v   = layers.Dense(1)
            self.a_h = layers.Dense(latent_a_dim, activation=activation)
            self.a   = layers.Dense(num_actions)

    def call(self, latent, training=None, mask=None):
        v = self.v(self.v_h(latent))
        a = self.a(self.a_h(latent))
        a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
        q = v + a - a_mean
        return q

    def reset_noise(self):
        if self.noise_std_init > 0:
            self.v_h.reset_noise()
            self.v.reset_noise()
            self.a_h.reset_noise()
            self.a.reset_noise()

    def set_noisy(self, active):
        if self.noise_std_init > 0:
            self.v_h.set_noisy(active)
            self.v.set_noisy(active)
            self.a_h.set_noisy(active)
            self.a.set_noisy(active)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_actions": self.num_actions,
            "latent_v_dim": self.latent_v_dim,
            "latent_a_dim": self.latent_a_dim,
            "noise_std_init": self.noise_std_init,
            "activation": self.activation,
        })
        return config

def build_model(num_actions, latent_dim=512, noise_std_init=0.5, dropout_rate=0.15):
    image_input = layers.Input(shape=(128, 128, 3), name='image_input')
    bird_input = layers.Input(shape=(5,), name='bird_input')

    stem = ClassicConvStem(latent_dim, dropout_rate=dropout_rate)
    latent = stem([image_input, bird_input])

    head = DuelingQHead(num_actions, noise_std_init=noise_std_init)
    q_values = head(latent)

    model = models.Model(inputs=[image_input, bird_input], outputs=q_values)
    return model
