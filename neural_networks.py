# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 00:25:27 2019

@author: 俊男
"""

# In[] Import Area
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# In[] create_seq_model(): A function generating a sequential neural network model
def create_seq_model(nodes=[], weight_init="glorot_normal",
                     hidden_activation="relu", opt_name="adam", metric_list=["acc"],
                     output_activation="softmax", loss_name="categorical_crossentropy"):
	# Check the version of Keras
	keras_version = tf.keras.__version__

	# Create Sequential Model
	model = Sequential()

	if nodes != []:
		# Create Input Layer
		if keras_version.startswith("2."):
			model.add(layers.InputLayer(input_shape=nodes[0], name="input"))
		elif keras_version.startswith("3."):
			model.add(layers.Input(shape=(nodes[0],), name="input"))
		else:
			raise ValueError(f"Unsupported Keras version: {keras_version}, please use 2.x or 3.x instead.")

		# Create Hidden Layers
		for i in range(1, len(nodes)-1):
			hidden_name = "hidden_{}".format(i)
			model.add(layers.Dense(units=nodes[i], kernel_initializer=weight_init, activation=hidden_activation, name=hidden_name))

		# Create Output Layers
		model.add(layers.Dense(units=nodes[-1], kernel_initializer=weight_init, activation=output_activation, name="output"))

		# Compile Neural Network
		model.compile(optimizer=opt_name, loss=loss_name, metrics=metric_list)

	# Return built model
	return model