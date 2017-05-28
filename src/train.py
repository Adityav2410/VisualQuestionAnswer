import numpy as np
import pandas as pd
import tensorflow as tf
import data_loader
import sys
import os


def initializeWeights(net):

	net.sess.run(tf.global_variables_initializer())

	ckpt = tf.train.get_checkpoint_state(model_dir)

	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and load_model:
		print "Restoring Model"
		net.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
	else:
		print "Initializing Model"
		net.sess.run(tf.global_variables_initializer())




def trainNetwork(net, num_epochs):

	# Get handle for vgg model
	initializeWeights(net)
	
	vgg,images = data_loader.getVGGhandle()

	# Parse all the vqa question informations
	qa_data = data_loader.load_questions_answers(net.datapath)
	data_validation =      qa_data['validation']
	data_training =        qa_data['training']
	question_vocab =       qa_data['question_vocab']
	answer_vocab =         qa_data['answer_vocab']
	question_input_dim =   len(question_vocab)
	answer_out_dim =       len(answer_vocab)

	num_training_data = len(data_training)
	nIter = num_training_data // net.batchSize

	# Prepare data generator which will be used for training the network
	train_data_generator = data_loader.getNextBatch(net.sess ,vgg,images,data_training,question_vocab,answer_vocab,os.path.join(net.image_base_path,'train2014'), batchSize = net.batchSize, purpose='train')
	valid_data_generator = data_loader.getNextBatch(net.sess ,vgg,images,data_validation,question_vocab,answer_vocab,os.path.join(net.image_base_path,'val2014'), batchSize = net.batchSize, purpose='val')


	# Generate data in batches:
	# batch_question : [batchSize = 32, maxQuestionLength=22, questionVocabDim = 15xxx]
	# batch_answer   : [batchSize = 32, answer_vocab = 1000]
	# batch_image_id : [batchSize = 32, 'filename of all the images in the batch' -> ['487025', '487025', '78077' ...... ] ]
	# batch_features : [batchSize = 32, cnnHeight=14, cnnWidth=14, featureDim = 512]
	# batch_question,batch_answer,batch_image_id,batch_features = train_data_generator.next()

	prevLoss = sys.maxint
	batchCount = -1
	for i in range( num_epochs):
		for iter in range(nIter):
			batchCount += 1
			batch_question,batch_answer,batch_image_id,batch_features = train_data_generator.next()			

			if( batchCount%1 == 0):
				[curr_train_loss, curr_train_acc] = net.sess.run([net.cross_entropy, net.accuracy] , feed_dict = { 	net.qs_ip  : batch_question ,				\
																													net.ans_ip : batch_answer 	, 				\
																													net.cnn_ip : batch_features } )				

				valid_batch_question,valid_batch_answer,valid_batch_image_id,valid_batch_features = valid_data_generator.next()
				[curr_valid_loss, curr_valid_acc] = net.sess.run([net.cross_entropy, net.accuracy] , feed_dict = { 	net.qs_ip  : valid_batch_question ,   		\
																													net.ans_ip : valid_batch_answer   , 		\
																													net.cnn_ip : valid_batch_features } )		
				print "Batch:%d \t, TrainLoss: %.2f \t TrainAccuracy: %.2f \t, ValidLoss:%.2f \t ValidAccuracy:%.2f " % (iter,curr_train_loss,curr_train_acc,curr_valid_loss,curr_valid_acc)

			# train the batch
			net.sess.run( net.train_step, feed_dict = { net.qs_ip  : batch_question ,
														net.ans_ip : batch_answer , 
														net.cnn_ip : batch_features } )






