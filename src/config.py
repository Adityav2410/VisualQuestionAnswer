import os
import sys
import numpy as np
import tensorflow as tf

class Config:
	def __init__(self):

		self.datapath = '../data/'
		self.image_base_path = '../data/'
		self.model_dir = './checkpoints/'
		self.embedd_weight_path = '../weights/embedd_weight.npy'


		self.num_epochs = 1
		self.batchSize = 1
		self.task_to_perform = 'train'

		self.verbose = False
