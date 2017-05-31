import numpy as np
import pandas as pd
import tensorflow as tf
import data_loader
import sys
import os

# -*- coding: utf-8 -*-


def trainNetwork(sess, net, num_epochs, C, saver_all):
# -*- coding: utf-8 -*-
	# Get handle for vgg model
	vgg,images = data_loader.getVGGhandle()

	# Parse all the vqa question informations
	qa_data = data_loader.load_questions_answers(C.datapath)
	data_validation =      qa_data['validation']
	data_training =        qa_data['training']
	question_vocab =       qa_data['question_vocab']
	answer_vocab =         qa_data['answer_vocab']
	question_input_dim =   len(question_vocab)
	answer_out_dim =       len(answer_vocab)

	num_training_data = len(data_training)
	nIter = num_training_data // net.batchSize

	# Prepare data generator which will be used for training the network
	train_data_generator = data_loader.getNextBatch(sess ,vgg,images, data_training,  question_vocab, answer_vocab,os.path.join(C.image_base_path,'train2014'), batchSize = C.batchSize, purpose='train')
	valid_data_generator = data_loader.getNextBatch(sess ,vgg,images, data_validation,question_vocab, answer_vocab,os.path.join(C.image_base_path,'val2014'),   batchSize = C.batchSize, purpose='val')

	# global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

	# Generate data in batches:
	# batch_question : [batchSize = 32, maxQuestionLength=22, questionVocabDim = 15xxx]
	# batch_answer   : [batchSize = 32, answer_vocab = 1000]
	# batch_image_id : [batchSize = 32, 'filename of all the images in the batch' -> ['487025', '487025', '78077' ...... ] ]
	# batch_features : [batchSize = 32, cnnHeight=14, cnnWidth=14, featureDim = 512]
	# batch_question,batch_answer,batch_image_id,batch_features = train_data_generator.next()

	batch_question,batch_answer,batch_image_id,batch_features = train_data_generator.next()			
	prev_loss = sess.run(net.cross_entropy, feed_dict = { 		net.qs_ip  : batch_question ,	\
																net.ans_ip : batch_answer 	, 	\
																net.cnn_ip : batch_features })
	print("Initial Loss: ", prev_loss)
	# global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
	# sess.run(tf.initialize_variables([global_step]))
	batchCount = -1
	print("Training network")
	for i in range( num_epochs):
		for iter in range(nIter):
			batchCount += 1
			batch_question,batch_answer,batch_image_id,batch_features = train_data_generator.next()			

			if( batchCount%100 == 0):
				[curr_train_loss, curr_train_acc , train_summary] = sess.run([net.cross_entropy, net.accuracy ,net.summary_op] , 
																				feed_dict = { 	net.qs_ip  : batch_question ,				\
																								net.ans_ip : batch_answer 	, 				\
																								net.cnn_ip : batch_features } )				

				net.writer.add_summary(train_summary)

				valid_batch_question,valid_batch_answer,valid_batch_image_id,valid_batch_features = valid_data_generator.next()
				[curr_valid_loss, curr_valid_acc, valid_summary ] = sess.run([net.cross_entropy, net.accuracy ,net.summary_op] , 
																				feed_dict = {	net.qs_ip  : valid_batch_question ,   		\
																								net.ans_ip : valid_batch_answer   , 		\
																								net.cnn_ip : valid_batch_features } )		

				if(curr_train_loss < prev_loss):
					prev_loss = curr_train_loss
					print("Loss decreased from %.4f to %.4f"%(prev_loss,curr_train_loss))
					print("Saving session")
					saver_all.save(sess,'checkpoints/vqa',global_step=net.global_step)

				print "Batch:%d \t, TrainLoss: %.2f \t TrainAccuracy: %.2f \t, ValidLoss:%.2f \t ValidAccuracy:%.2f " % (iter,curr_train_loss,curr_train_acc,curr_valid_loss,curr_valid_acc)
				
			# train the batch
			sess.run( net.train_step, feed_dict = 	{ 	net.qs_ip  : batch_question ,
														net.ans_ip : batch_answer , 
														net.cnn_ip : batch_features } )






