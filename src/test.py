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


import argparse
import os
import sys
import tensorflow as tf
from model import *
from train import trainNetwork
from config import Config
#from test import testNetwork
from pdb import set_trace as bp
import utils

# -*- coding: utf-8 -*-

# get answer for a given question
# def getAnswerToQuestion(sess,image_file_path, vgg_handle, image_placeholder, question,vocab, reverse_answer_vocab, numAnswer = 3):


# 	# Read image and get its feature map
# 	img 			= 	skimage.io.imread(image_file_path)
# 	if( len(img.shape) < 3 or img.shape[2] < 3 ) :
# 			continue		

# 	imgFeatures 	= 	data_loader.getImageFeatures(sess,vgg_handle, image_placeholder, img)
# 	[q_vec,ans_vec] = 	data_loader.QnAinVectorNotation(question,answer, vocab)	


	
# 	predicted_prob, attn_map_t0,attn_map_t8,attn_map_t17, attn_map_t19,attn_map_t21 = sess.run(net.ans_op,net.attn_map_t0 ,
# 																								net.attn_map_t8,net.attn_map_t17,net.attn_map_t19,net.attn_map_t21,
# 		  																						feed_dict = { net.qs_ip  : q_vec ,
# 		  																									  net.cnn_ip : imgFeatures })

# 	[top_predicted_answer,predicted_answer_prob] = parse_predicted_probabilities(predicted_prob, reverse_answer_vocab, numAnswer)
# 	attn_map 		= 	[	attn_map_t0[0],	attn_map_t8[0],	attn_map_t17[0],	attn_map_t19[0],	attn_map_t21[0]	]

# 	return( top_predicted_answer, predicted_answer_prob, attn_map )



def visualizeNetwork(sess, net, C):
# -*- coding: utf-8 -*-
	# Get handle for vgg model
	vgg,images = data_loader.getVGGhandle()

	# Parse all the vqa question informations
	qa_data = data_loader.load_questions_answers(C.datapath)
	data_validation =      qa_data['validation']
	data_training =        qa_data['training']
	question_vocab =       qa_data['question_vocab']
	answer_vocab =         qa_data['answer_vocab']
	reverse_answer_vocab = data_loader.get_reverse_vocab(answer_vocab)
	reverse_quest_vocab  =	data_loader.get_reverse_vocab(question_vocab)

	train_data_path = os.path.join(C.image_base_path,'train2014')
	val_data_path 	= os.path.join(C.image_base_path,'val2014')
	train_data_generator = data_loader.getNextBatch(sess ,vgg,images, data_training,  question_vocab, answer_vocab,train_data_path, batchSize = 1, purpose='train')
	valid_data_generator = data_loader.getNextBatch(sess ,vgg,images, data_validation,question_vocab, answer_vocab,val_data_path, batchSize = 1, purpose='val')

	save_path = '../vizQnA/'

	
	for i in range(C.max_visualize):
		batch_question,batch_answer,batch_image_id,batch_features = train_data_generator.next()	
		image_path = train_data_path

		image_save_dir = os.path.join( save_path, batch_image_id[0])
		utils.make_dir(image_save_dir)

		[predicted_prob, attn_map_t0,attn_map_t8,attn_map_t17, attn_map_t19,attn_map_t21 ]= sess.run([net.ans_op_prob,net.attn_map_t0 ,net.attn_map_t8,				\
																										net.attn_map_t17,net.attn_map_t19,net.attn_map_t21]	, 	\
		  																							feed_dict = { net.qs_ip  : batch_question ,					\
		  																										  net.cnn_ip : batch_features })


		[top_predicted_answer,predicted_answer_prob] = utils.parse_predicted_probabilities(predicted_prob[0], C.numAnswer)
		attn_map 		= 	[	attn_map_t0[0],	attn_map_t8[0],	attn_map_t17[0],	attn_map_t19[0],	attn_map_t21[0]	]

		utils.process_results( top_predicted_answer, predicted_answer_prob, attn_map, image_path, batch_question[0], batch_answer[0], \
								batch_image_id[0],  image_save_dir, reverse_quest_vocab, reverse_answer_vocab, purpose='train' )




def testNetwork(sess,net,C):
	# bp()
	C.max_visualize = 100
	visualizeNetwork(sess,net,C)


