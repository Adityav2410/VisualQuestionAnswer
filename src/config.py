import numpy as np
import os

class Config:

	def __init__(self):
		self.batchSize = 10
		self.datapath = '/home/adityav/ADITYA/Project/Cogs_Image_Caption/VisualQuestionAnswer/data/'
		self.imagePath = os.path.join(self.datapath + 'train2014/')
		self.embeddSize = 256
		

