import numpy as np
import pandas as pd
import tensorflow as tf
import data_loader
import sys
import os
from datetime import datetime
import time
import data_loader
import skimage.io


# -*- coding: utf-8 -*-

def getAnswerToQuestion(sess,image_file, vgg_handle, image_placeholder, question,answer,vocab, numAnswer = 3):


	# Read image and get its feature map
	img 		= 	skimage.io.imread(image_file)
	imgFeatures = 	data_loader.getImageFeatures(sess,vgg_handle, image_placeholder, img)


	# Parse all the vqa question informations
	vocab 	= 	data_loader.load_vocab(C.vocab_file)
	

	 = sess.run(net.cross_entropy, feed_dict = { 		net.qs_ip  : batch_question ,	\
																net.ans_ip : batch_answer 	, 	\
																net.cnn_ip : batch_features })



	 
		# print img.shape

		if( len(img.shape) < 3 or img.shape[2] < 3 ) :
			continue		

		batch_id.append(str(qa_id))
		# Image is valid - Get features of the image
		




def testNetwork(sess, net, num_epochs, C):
# -*- coding: utf-8 -*-
	# Get handle for vgg model
	vgg,images = data_loader.getVGGhandle()

	# Parse all the vqa question informations
	vocab = load_vocab(C.vocab_file)





	# global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


	# global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
	# sess.run(tf.initialize_variables([global_step]))
	batchCount = -1
	log_filename = './log_dir/train_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'
	fHandle = open( log_filename, 'w')
	print("Writing log to file: ", log_filename)

	print("Training network\n")
	print("Initial Loss: ", prev_loss)
	print "Number of epochs:%d , \t Iteration per epoch:%d" % ( num_epochs, nIter)
	fHandle.write("Training Network\n")

	fHandle.write("Initial Loss: \n" % (prev_loss))

	start_time = time.time()
	
	for epoch in range( num_epochs):
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
					print("Loss decreased from %.4f to %.4f"%(prev_loss,curr_train_loss))
					print("Saving session")
					fHandle.write("Loss decreased from %.4f to %.4f"%(prev_loss,curr_train_loss))
					saver_all.save(sess,'checkpoints/vqa',global_step=net.global_step)
					prev_loss = curr_train_loss
				print "Epoc:%d/%d_Iter:%d/%d,  TrainLoss:%.2f  TrainAccuracy:%.2f,  ValidLoss:%.2f  ValidAccuracy:%.2f  Elapsed time: %d" % (epoch,num_epochs,iter,nIter,curr_train_loss,curr_train_acc*100,curr_valid_loss,curr_valid_acc*100,time.time()-start_time)
				fHandle.write("Epoc:%d/%d_Iter:%d/%d \t, TrainLoss: %.2f \t TrainAccuracy: %.2f \t, ValidLoss:%.2f \t ValidAccuracy:%.2f \t Elapsed time: %d\n" % (epoch,num_epochs,iter,nIter,curr_train_loss,curr_train_acc*100,curr_valid_loss,curr_valid_acc*100,time.time()-start_time))
				start_time = time.time()
			# train the batch
			sess.run( net.train_step, feed_dict = 	{ 	net.qs_ip  : batch_question ,
														net.ans_ip : batch_answer , 
														net.cnn_ip : batch_features } )
			# net.print_variables()
	fHandle.close()






