#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

"""
Neural Networkの構成、出力の提供を行うモジュールを定義

Dimensions: 1D
version: TensorFlow 2.xx系
"""

class Network:
    def __init__(self, n_site, n_boson, hidden_units=[40,40]):
        self.n_site = n_site
        self.n_boson = n_boson
        self.hidden_units = hidden_units

        self.__init_model()

    def __init_model(self):
        inputs = layers.Input(
            shape=(self.n_site,) ,
            name='Input_Layer',
            dtype=tf.float32,
        )

        for n_layer in range(len(self.hidden_units)):
            layer_hidd = layers.Dense(
                units=self.hidden_units[n_layer],
                activation='relu',
                name='Layer{}'.format(n_layer+1)
            )
            if n_layer == 0:
                x = layer_hidd(inputs)
            else:
                x = layer_hidd(x)

        layer_out = layers.Dense(
            units=1,
            activation='relu',
            name='Output_Layer',
        )
        log_psi = layer_out(x)

        self.__model = CustomModel(
            inputs=inputs,
            outputs=log_psi,
            name='model_sites{}_boson{}'.format(self.n_site, self.n_boson),
        )

        self.__compile()

    def __compile(self):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
        )

        self.__model.compile(
            optimizer=optimizer,
        )

    def summary(self):
        return self.__model.summary()

    def fit(self, num, eloc):
        self.__model.fit(num, eloc)

    def output(self, num):
        return self.__model.predict(num)


class CustomModel(keras.Model):
    def train_step(self, data):
        num, eloc = data # model.fit(num, eloc)
        eloc_mean = tf.reduce_mean(eloc)

        with tf.GradientTape() as tape:
            log_psi = self(num)
            loss = tf.reduce_mean(log_psi*(eloc - eloc_mean))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"eloc_mean": eloc_mean}
