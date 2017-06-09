# The code for parsing the question answers has been copied from the github account of Paarth Neekhara. However they have been modified a bit for using as per our specification
# https://github.com/paarthneekhara/neural-vqa-tensorflow
import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle
from itertools import cycle
import vgg16.vgg16 as vgg_model
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import os 
import random
import re
import time
from pdb import set_trace as bp

def load_questions_answers(data_dir):
	
	questions = None
	answers = None
	
	t_q_json_file = join(data_dir, 'MultipleChoice_mscoco_train2014_questions.json')
	t_a_json_file = join(data_dir, 'mscoco_train2014_annotations.json')

	v_q_json_file = join(data_dir, 'MultipleChoice_mscoco_val2014_questions.json')
	v_a_json_file = join(data_dir, 'mscoco_val2014_annotations.json')
	qa_data_file = join(data_dir, 'qa_data_file.pkl')
	vocab_file = join(data_dir, 'vocab_file.pkl')

	# IF ALREADY EXTRACTED
	if isfile(qa_data_file):
		with open(qa_data_file) as f:
			data = pickle.load(f)
			return data

	print "Loading Training questions"
	with open(t_q_json_file) as f:
		t_questions = json.loads(f.read())
	
	print "Loading Training anwers"
	with open(t_a_json_file) as f:
		t_answers = json.loads(f.read())

	print "Loading Val questions"
	with open(v_q_json_file) as f:
		v_questions = json.loads(f.read())
	
	print "Loading Val answers"
	with open(v_a_json_file) as f:
		v_answers = json.loads(f.read())

	
	print "Ans", len(t_answers['annotations']), len(v_answers['annotations'])
	print "Qu", len(t_questions['questions']), len(v_questions['questions'])

	answers = t_answers['annotations'] + v_answers['annotations']
	questions = t_questions['questions'] + v_questions['questions']
	
	answer_vocab = make_answer_vocab(answers)
	question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab)
	print "Max Question Length", max_question_length
	word_regex = re.compile(r'\w+')
	training_data = []
	for i,question in enumerate( t_questions['questions']):
		ans = t_answers['annotations'][i]['multiple_choice_answer']
		if ans in answer_vocab:
			training_data.append({
				'image_id' : t_answers['annotations'][i]['image_id'],
				'question' : np.zeros(max_question_length),
				'answer' : answer_vocab[ans]
				})
			question_words = re.findall(word_regex, question['question'])

			base = max_question_length - len(question_words)
			for i in range(0, len(question_words)):
				training_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]

	print "Training Data", len(training_data)
	val_data = []
	for i,question in enumerate( v_questions['questions']):
		ans = v_answers['annotations'][i]['multiple_choice_answer']
		if ans in answer_vocab:
			val_data.append({
				'image_id' : v_answers['annotations'][i]['image_id'],
				'question' : np.zeros(max_question_length),
				'answer' : answer_vocab[ans]
				})
			question_words = re.findall(word_regex, question['question'])

			base = max_question_length - len(question_words)
			for i in range(0, len(question_words)):
				val_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]

	print "Validation Data", len(val_data)

	data = {
		'training' : training_data,
		'validation' : val_data,
		'answer_vocab' : answer_vocab,
		'question_vocab' : question_vocab,
		'max_question_length' : max_question_length
	}

	print "Saving qa_data"
	with open(qa_data_file, 'wb') as f:
		pickle.dump(data, f)

	with open(vocab_file, 'wb') as f:
		vocab_data = {
			'answer_vocab' : data['answer_vocab'],
			'question_vocab' : data['question_vocab'],
			'max_question_length' : data['max_question_length']
		}
		pickle.dump(vocab_data, f)

	return data

def get_question_answer_vocab(data_dir):
	vocab_file = join(data_dir, 'vocab_file.pkl')
	vocab_data = pickle.load(open(vocab_file))
	return vocab_data

def make_answer_vocab(answers):
	top_n = 1000
	answer_frequency = {} 
	for annotation in answers:
		answer = annotation['multiple_choice_answer']
		if answer in answer_frequency:
			answer_frequency[answer] += 1
		else:
			answer_frequency[answer] = 1

	answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.iteritems()]
	answer_frequency_tuples.sort()
	answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

	answer_vocab = {}
	for i, ans_freq in enumerate(answer_frequency_tuples):
		# print i, ans_freq
		ans = ans_freq[1]
		answer_vocab[ans] = i

	answer_vocab['UNK'] = top_n - 1
	return answer_vocab


def make_questions_vocab(questions, answers, answer_vocab):
	word_regex = re.compile(r'\w+')
	question_frequency = {}

	max_question_length = 0
	for i,question in enumerate(questions):
		ans = answers[i]['multiple_choice_answer']
		count = 0
		if ans in answer_vocab:
			question_words = re.findall(word_regex, question['question'])
			for qw in question_words:
				if qw in question_frequency:
					question_frequency[qw] += 1
				else:
					question_frequency[qw] = 1
				count += 1
		if count > max_question_length:
			max_question_length = count


	qw_freq_threhold = 0
	qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.iteritems()]
	# qw_tuples.sort()

	qw_vocab = {}
	for i, qw_freq in enumerate(qw_tuples):
		frequency = -qw_freq[0]
		qw = qw_freq[1]
		# print frequency, qw
		if frequency > qw_freq_threhold:
			# +1 for accounting the zero padding for batc training
			qw_vocab[qw] = i + 1
		else:
			break

	qw_vocab['UNK'] = len(qw_vocab) + 1

	return qw_vocab, max_question_length

def getImage(datapath, imageID, purpose='train'):
	name_3 = str(imageID)
	name_2 = '0' * (12-len(name_3))
	name_1 = 'COCO_' + purpose + '2014_'
	fileName = name_1 + name_2 + name_3 + '.jpg'
	filepath = join(datapath,fileName)
	img = skimage.io.imread(filepath)
	return(img)

def getImageFeatures(sess, vgg, images ,img ):
    # load image
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    img_reshape = 	resized_img.reshape((1, 224, 224, 3))
    img_feature = sess.run(vgg.pool4, feed_dict={images:img_reshape})
    return(img_feature)


def getNextBatch(sess, vgg, images, qa_data, question_vocab, answer_vocab, datapath, batchSize=32, purpose='train'):
	
	currIndex = 0;
	questionVocabSize = len(question_vocab.keys())
	answerVocabSize = len(answer_vocab.keys())
	question_length = qa_data[0]['question'].shape[0]

	# quest_oneHot = np.zeros((1,questionVocabSize,question_length))
	ans_oneHot = np.zeros((1,answerVocabSize))
	# batch_quest = np.zeros((0,question_length,questionVocabSize))
	batch_quest = np.zeros((0,question_length))
	batch_ans   = np.zeros((0,answerVocabSize))
	batchFeatures = np.zeros((0,14,14,512))
	batch_id = []

	for iter in cycle(qa_data):
		qa_id    =  iter['image_id']
		qa_ans   =  iter['answer']
		qa_quest =  iter['question']

		# Checking if the question is for a valid image or not
		img = getImage(datapath, qa_id, purpose )
		# print img.shape

		if( len(img.shape) < 3 or img.shape[2] < 3 ) :
			continue		

		batch_id.append(str(qa_id))
		# Image is valid - Get features of the image
		imgFeatures = getImageFeatures(sess,vgg, images, img)
		batchFeatures = np.concatenate((batchFeatures,imgFeatures),0)

		# Convert the question and answer in One Hot format
		currIndex = currIndex + 1

		ans_oneHot = np.zeros((1,answerVocabSize))
		# quest_oneHot = np.zeros((1,question_length,questionVocabSize))
		# for i in range(qa_quest.shape[0]):
			# quest_oneHot[ 0,i,int(qa_quest[i]) ] = 1
		ans_oneHot[0,qa_ans] = 1
		# print(qa_quest)
		# print(qa_quest.shape)
		# print(batch_quest.shape)
		# bp()
		# Concat all the question in the batch
		batch_quest = np.concatenate((batch_quest,np.reshape(qa_quest,(1,-1))),0)
		batch_ans   = np.concatenate((batch_ans  ,ans_oneHot),0)

		# print("Question shape:", batch_quest.shape)
		# print("Answer shape:", batch_quest.shape)
		# print("ID shape:", batch_id)
		# print("Feature shape:", batch_quest.shape)

		if currIndex == batchSize:
			yield np.copy(batch_quest),np.copy(batch_ans),batch_id[:],np.copy(batchFeatures) 
			batch_id = []
			batch_quest 	= np.zeros((0,question_length))
			batch_ans   	= np.zeros((0,answerVocabSize))
			batchFeatures 	= np.zeros((0,14,14,512))
			currIndex 		= 0



def getVGGhandle():
	images = tf.placeholder("float", [None, 224, 224, 3])
	vgg = vgg_model.Vgg16()
	with tf.name_scope("content_vgg"):
		vgg.build(images)
	return(vgg,images)




def load_vocab(vocab_file):
	if isfile(vocab_file):
		with open(vocab_file) as f:
			vocab = pickle.load(f)
			return(vocab)
	else:
		print("Incorrect vocab file")
		return(None)



def QnAinVectorNotation(question,answer, vocab):
    q_vocab = vocab['question_vocab']
    ans_vocab = vocab['answer_vocab']
    max_question_length = vocab['max_question_length']

    q_vec = np.zeros(max_question_length)

    # convert question in vector notation
    word_regex = re.compile(r'\w+')
    question_words = re.findall(word_regex, question)
    base = max_question_length - len(question_words)
    for i in range(0, len(question_words)):
        q_vec[base + i] = q_vocab[ question_words[i] ]

    # convert answer in vector notation
    ans_vec = ans_vocab.get(answer)

    return(q_vec, ans_vec)


