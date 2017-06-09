import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import *

from pdb import set_trace as bp
from utils import gen_plot

class Model:

	def __init__(self,sess=None,batchSize=32, C = None):


		#### Variables related to image basic configuration
		self.sess = sess
		self.batchSize = batchSize

		#### Variables related to basic vocab & QnA
		self.question_dim = 15183
		self.answer_dim = 1000
		self.max_qs_len = 22

		#### Variables related to RNN
		self.rnn_ip_dim = 1024
		self.rnn1_hidden_dim  = 512
		self.rnn2_hidden_dim  = 128
		
		#### Variables related to CNN feature map
		self.cnn_dim = 512
		self.cnn_width = 14
		self.cnn_height = 14
		self.word_embedd = np.load(C.embedd_weight_path)

		#### Basic hyper-parameters ######################
		self.lr = 0.01
	def build_network(self):

		self.cnn_ip = tf.placeholder(tf.float32,shape=[self.batchSize,self.cnn_width,self.cnn_height,self.cnn_dim])
		# self.qs_ip = tf.placeholder(tf.float32,shape=[self.batchSize,self.max_qs_len,self.question_dim])
		self.qs_ip = tf.placeholder(tf.int32,shape=[self.batchSize,self.max_qs_len])
		self.ans_ip = tf.placeholder(tf.float32,shape=[self.batchSize,self.answer_dim])
		#self.rnn_input = tf.placeholder(tf.float32,shape=[self.max_qs_len,self.batchSize,self.question_dim])
		self.rnn_input_encoded = tf.zeros([self.batchSize, self.rnn_ip_dim//2], name="qs_embedded")
		
		self.rnn_attn_vec = tf.zeros([self.batchSize,self.cnn_dim])
		self.rnn_attn_vec = tf.cast(self.rnn_attn_vec, tf.float32)

		self.attn_prob    = tf.zeros([self.batchSize,self.cnn_height,self.cnn_width,1])
		self.attn_prob    = tf.cast(self.attn_prob, tf.float32)

		self.attn_vec = tf.zeros([self.batchSize,self.rnn_ip_dim//2])
		self.attn_vec = tf.cast(self.attn_vec,tf.float32)

		# self.dummy_check = tf.zeros([self.batchSize,self.rnn_ip_dim])
		# self.dummy_check = tf.cast(self.dummy_check,tf.float32)
		
		# Counter variable - counts the number of times train step has been called
		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
		
		with tf.variable_scope('vqa_weights'):
			self.qs_to_ip_w = tf.Variable(tf.constant(self.word_embedd),name='qs_to_rnn_w', trainable=False)
			self.ip_to_rnn1_w = tf.Variable(tf.random_normal([self.rnn_ip_dim,self.rnn1_hidden_dim]),name='ip_to_rnn1_w')
			self.rnn_to_attn_w = tf.Variable(tf.random_normal([self.rnn1_hidden_dim,self.cnn_dim]),name='rnn_to_attn_w')

			self.cnn_attn_to_ip_w  = tf.Variable(tf.random_normal([self.cnn_dim,self.rnn_ip_dim//2]),name='cnn_attn_to_ip_w')
			self.rnn2_to_ans_w  = tf.Variable(tf.random_normal([self.rnn2_hidden_dim,self.answer_dim]),name='rnn2_to_ans_w')

			self.qs_to_ip_b = tf.Variable(tf.zeros([self.rnn_ip_dim//2]),name='qs_to_rnn_b')
			self.ip_to_rnn1_b = tf.Variable(tf.zeros([self.rnn1_hidden_dim]),name='ip_to_rnn1_b')
			self.rnn_to_attn_b = tf.Variable(tf.zeros([self.cnn_dim]),name='rnn_to_attn_b')

			self.cnn_attn_to_ip_b = tf.Variable(tf.zeros([self.rnn_ip_dim//2]),name='cnn_attn_to_ip_b')
			self.rnn2_to_ans_b  = tf.Variable(tf.zeros([self.answer_dim]),name='rnn2_to_ans_b')
		
		self.rnn1_cell = BasicLSTMCell(self.rnn1_hidden_dim,state_is_tuple=False)
		self.rnn2_cell = BasicLSTMCell(self.rnn2_hidden_dim,state_is_tuple=False)
		
		self.hidden_state_rnn1 = tf.tile(tf.get_variable(
			'rnn1_states', [1,2*self.rnn1_hidden_dim],
			tf.float32,tf.constant_initializer(value=0,dtype=tf.float32),trainable=False),
			[self.batchSize,1])

		self.hidden_state_rnn2 = tf.tile(tf.get_variable(
			'rnn2_states', [1,2*self.rnn2_hidden_dim],
			tf.float32,tf.constant_initializer(value=0,dtype=tf.float32),trainable=False),
			[self.batchSize,1])


		self.op_rnn1 = tf.tile( tf.get_variable(
			'rnn1_op',[1,self.rnn1_hidden_dim],
			tf.float32,tf.constant_initializer(value=0,dtype=tf.float32),trainable=False),
			[self.batchSize,1]	)

		self.op_rnn2 = tf.tile( tf.get_variable(
			'rnn2_op',[1,self.rnn2_hidden_dim],
			tf.float32,tf.constant_initializer(value=0,dtype=tf.float32),trainable=False),
			[self.batchSize,1]	)

		# self.rnn_input = tf.transpose(self.qs_ip, [1,0,2])
		self.rnn_input = tf.transpose(self.qs_ip, [1,0])

		#bp()
		def fn(ip):
			
			# self.rnn_input_encoded = tf.matmul(ip,self.qs_to_ip_w) + self.qs_to_ip_b
			self.rnn_input_encoded = tf.nn.embedding_lookup(self.qs_to_ip_w, ip)
			self.rnn_ip = tf.concat([self.rnn_input_encoded,self.attn_vec],1)
			
			with tf.variable_scope('rnn1_weights') as scope:
				self.op_rnn1,self.hidden_state_rnn1 = self.rnn1_cell(self.rnn_ip,self.hidden_state_rnn1)
				scope.reuse_variables()
			with tf.variable_scope('rnn2_weights'):
				self.op_rnn2,self.hidden_state_rnn2 = self.rnn2_cell(self.op_rnn1,self.hidden_state_rnn2)
				scope.reuse_variables()

			self.rnn_attn_vec = tf.matmul(self.op_rnn1,self.rnn_to_attn_w, name='rnn_to_att_vec') + self.rnn_to_attn_b
			self.rnn_attn_vec = tf.reshape( self.rnn_attn_vec, shape=[self.batchSize,1,1,self.cnn_dim])

			self.attn_prob  = tf.multiply(self.cnn_ip, self.rnn_attn_vec,name='attn_prob')
			self.attn_prob  = tf.reduce_sum(self.attn_prob,axis=3,name='sum1')
			self.attn_prob  = tf.reshape( self.attn_prob,shape=[self.batchSize,-1])
			self.attn_prob  = tf.nn.softmax(self.attn_prob)
			self.attn_prob  = tf.reshape(self.attn_prob,shape = [self.batchSize,self.cnn_height,self.cnn_width,1])

			self.attn_vec   = tf.multiply(self.cnn_ip,self.attn_prob,name='attn_vector')
			self.attn_vec   = tf.reduce_sum(self.attn_vec,axis = 2,name='sum2')
			self.attn_vec   = tf.reduce_sum(self.attn_vec,axis = 1,name='sum3')

			self.attn_map 	= tf.reshape(self.attn_prob,shape=[self.batchSize, self.cnn_width*self.cnn_height])
			return tf.concat([self.op_rnn2  ,self.attn_map],axis=1)


		
		fun = tf.make_template('fun', fn)
		self.res = tf.map_fn(fn,self.rnn_input,dtype=tf.float32)
		print self.res.get_shape()

		self.res_rnn2 = self.res[-1][:,:128]
		self.ans_op = tf.matmul( self.res_rnn2, self.rnn2_to_ans_w ) + self.rnn2_to_ans_b

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ans_ip, logits=self.ans_op))
		self.train_step = tf.train.MomentumOptimizer(learning_rate = self.lr, momentum = 0.9, use_nesterov=True).minimize(self.cross_entropy)
		#self.train_step = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(self.cross_entropy,global_step = self.global_step)
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

			tf.summary.histogram("histogram_loss",self.cross_entropy)
			tf.summary.histogram("true_ans",self.true_answer)
			tf.summary.histogram("predicted_ans",self.predicted_answer)
		
		
			tf.summary.image('attn_map_t0', tf.reshape(self.res[0][:,128:],shape=[self.batchSize,self.cnn_width,self.cnn_height,1]) )
			tf.summary.image('attn_map_t8', tf.reshape(self.res[8][:,128:],shape=[self.batchSize,self.cnn_width,self.cnn_height,1]) )
			tf.summary.image('attn_map_t17', tf.reshape(self.res[17][:,128:],shape=[self.batchSize,self.cnn_width,self.cnn_height,1]) )
			tf.summary.image('attn_map_t19', tf.reshape(self.res[19][:,128:],shape=[self.batchSize,self.cnn_width,self.cnn_height,1]) )
			tf.summary.image('attn_map_t21', tf.reshape(self.res[21][:,128:],shape=[self.batchSize,self.cnn_width,self.cnn_height,1]) )
			# tf.summary.image('normal_distribution', tf.random_normal(shape=[self.batchSize,self.cnn_width,self.cnn_height,1], dtype=tf.float32) )
			# tf.summary.image('true_vs_predicted_ans', self.answer_plot)
		
			# tf.summary.image('attention_map',self.res_attn_map)
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



