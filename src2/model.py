import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import *

from pdb import set_trace as bp
# from utils import gen_plot

class Model:

	def __init__(self,sess=None,batchSize=None, C = None):


		#### Variables related to image basic configuration
		self.sess = sess
		self.batchSize = batchSize

		#### Variables related to basic vocab & QnA
		self.question_dim = 15183
		self.answer_dim = 1000
		self.max_qs_len = 22

		#### Variables related to RNN
		self.rnn1_hidden_dim  = 512
		
		#### Variables related to CNN feature map
		self.cnn_dim 		= 512
		self.fc_dim 		= 4096
		self.fc_embedd_dim	= 512
		self.cnn_width 		= 14
		self.cnn_height 	= 14
		self.word_embedd 	= np.load(C.embedd_weight_path)
		self.word_embedd_dim= 512
		self.pre_op_dim 	= 128

		#### Basic hyper-parameters ######################
		self.lr = 0.01
	def build_network(self):

		self.cnn_ip = tf.placeholder(tf.float32,shape=[self.batchSize,self.cnn_width,self.cnn_height,self.cnn_dim])
		self.fc_ip	= tf.placeholder(tf.float32,shape=[self.batchSize,self.fc_dim])
		self.qs_ip = tf.placeholder(tf.int32,shape=[self.batchSize,self.max_qs_len])
		self.ans_ip = tf.placeholder(tf.float32,shape=[self.batchSize,self.answer_dim])

		self.qs_encoded = tf.zeros(dtype=tf.float32,shape=[self.batchSize, self.rnn1_hidden_dim], name="qs_embedded")
		self.rnn1_input = tf.zeros(dtype=tf.float32, shape=[self.max_qs_len,self.batchSize, self.word_embedd_dim])
		# Counter variable - counts the number of times train step has been called
		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
		
	
		with tf.variable_scope('vqa_weights'):
			self.qs_to_ip_w = tf.Variable(tf.constant(self.word_embedd),name='qs_to_rnn_w', trainable=False)
			self.rnn1_to_attn_w = tf.Variable(tf.random_normal([self.rnn1_hidden_dim,self.cnn_dim]),name='rnn_to_attn_w')
			self.rnn1_to_attn_b = tf.Variable(tf.random_normal([self.cnn_dim]),name='rnn_to_attn_b')
	
			self.fc_embedd_w	= tf.Variable(tf.random_normal([self.fc_dim, self.fc_embedd_dim]),name='fc_embedd_w')
			self.fc_embedd_b	= tf.Variable(tf.random_normal([self.fc_embedd_dim]),name='fc_embedd_b')

			self.pre_op_w 		= tf.Variable(tf.random_normal([self.cnn_dim + self.fc_embedd_dim + self.rnn1_hidden_dim, self.pre_op_dim]),name='pre_op_w')
			self.pre_op_b		= tf.Variable(tf.random_normal([self.pre_op_dim]),name='pre_op_b')

			self.op_w 			= tf.Variable(tf.random_normal([self.pre_op_dim,self.answer_dim]),name='op_w')
			self.op_b 			= tf.Variable(tf.random_normal([self.answer_dim]),name='op_b')
		# From input to pre-attention vector
		self.rnn1_cell = BasicLSTMCell(self.rnn1_hidden_dim,state_is_tuple=False)
		
		# Embedd question and compute pre-attention state

		q_embedd_list = []
		for i in range(self.max_qs_len):
			q_embedd_list.append(tf.nn.embedding_lookup(self.qs_to_ip_w, self.qs_ip[:,i]))
		self.rnn1_input = tf.stack(q_embedd_list)

		with tf.variable_scope('rnn1') as scope:
			[self.rnn1_outputs, self.rnn1_states] = static_rnn( cell = self.rnn1_cell,inputs = q_embedd_list , dtype = tf.float32)

		# get attention vector
		self.qs_encoded = self.rnn1_outputs[-1]
		self.rnn_attn_vec = tf.matmul( self.qs_encoded, self.rnn1_to_attn_w,name='rnn1_to_attn_vec') + self.rnn1_to_attn_b
		self.rnn_attn_vec = tf.reshape( self.rnn_attn_vec, shape=[-1,1,1,self.cnn_dim])

		self.attn_prob  = tf.multiply(self.cnn_ip, self.rnn_attn_vec,name='attn_prob')
		self.attn_prob  = tf.reduce_sum(self.attn_prob,axis=3,name='sum1')
		self.attn_prob  = tf.reshape( self.attn_prob,shape=[self.batchSize,-1])
		self.attn_prob  = tf.nn.softmax(self.attn_prob)
		self.attn_prob  = tf.reshape(self.attn_prob,shape = [self.batchSize,self.cnn_height,self.cnn_width,1])

		self.attn_vec   = tf.multiply(self.cnn_ip,self.attn_prob,name='attn_vector')
		self.attn_vec   = tf.reduce_sum(self.attn_vec,axis = 2,name='sum2')
		self.attn_vec   = tf.reduce_sum(self.attn_vec,axis = 1,name='sum3')

		# Embedded question with image attention: 
		self.embedded_image = tf.matmul(self.fc_ip, self.fc_embedd_w , name='fc_embedd') + self.fc_embedd_b
		self.ques_image	=	tf.concat([ self.rnn1_outputs[-1], self.attn_vec, self.embedded_image ],axis=1)

		self.pre_op 	= tf.matmul( self.ques_image, self.pre_op_w, name='pre_op_w') + self.pre_op_b
		self.ans_op 	= tf.matmul( self.pre_op, self.op_w ) + self.op_b
		self.ans_op_prob = tf.nn.softmax( self.ans_op)

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ans_ip, logits=self.ans_op))
		# self.train_step = tf.train.MomentumOptimizer(learning_rate = self.lr, momentum = 0.9, use_nesterov=True).minimize(self.cross_entropy)
		self.train_step = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(self.cross_entropy,global_step = self.global_step)
		# bp()

		self.true_answer 		=	tf.argmax(self.ans_ip,1)
		self.predicted_answer 	= 	tf.argmax(self.ans_op,1)
		
		self.correct_prediction = tf.equal(tf.argmax(self.ans_ip,1),tf.argmax(self.ans_op,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))

		self.summary_op = self._create_summaries()


	def _create_summaries(self):
		with tf.name_scope("summaries"):
			tf.summary.scalar("loss",self.cross_entropy)
			tf.summary.scalar("accuracy", self.accuracy)

			# self.answer_plot = gen_plot(data1 = self.predicted_answer, data2 = self.true_answer,
			# 							label1 = 'predicted Answer', label2 = 'true Answer', title = 'True vs Predicted answer' ) 

			# tf.summary.histogram("histogram_loss",self.cross_entropy)
			# tf.summary.histogram("true_ans",self.true_answer)
			# tf.summary.histogram("predicted_ans",self.predicted_answer)
		
			tf.summary.image('attn_map' , self.attn_prob  )

			summary_op = tf.summary.merge_all()
		return(summary_op)



	def write_tensorboard(self):
		self.writer = tf.summary.FileWriter('../logs', self.sess.graph)
		# self.writer.flush()



	def print_variables(self):
		params = tf.all_variables()
		print("Number of parameters in network: ", len(params))

		for param in params:
			print param.name,param.shape



