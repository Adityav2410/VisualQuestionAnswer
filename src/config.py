import os
import sys
import numpy as np
import tensorflow as tf

class Config:
	def __init__(self):
		self.batchSize = 32
		self.datapath = '/home/adityav/ADITYA/Project/Cogs_Image_Caption/VisualQuestionAnswer/data/'
		self.image_base_path = '/home/adityav/ADITYA/Project/Cogs_Image_Caption/VisualQuestionAnswer/data/'
		self.model_dir = 'checkpoints/'


		self.num_epochs = 10
		self.batchSize = 32
		self.task_to_perform = 'train'

		self.verbose = False
