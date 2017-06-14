import io
import matplotlib.pyplot as plt
import tensorflow as tf
import os, shutil
from pdb import set_trace as bp
import data_loader
import numpy as np


def gen_plot(data1 = [2,1] , data2= [1,2], label1 = 'trainData', label2='testData', title = 'train vs test'):
    
    """Create a pyplot plot and save to buffer."""
    legend1, = plt.plot(data1,label=label1)
    legend2, = plt.plot(data2,label=label2)
    plt.title(title)    
    buf = io.BytesIO()
    
    plt.savefig(buf, format='png')
    # plot_buf = buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels = 1)
    image = tf.expand_dims(image, 0)

    return image


# image = gen_plot()



def parse_predicted_probabilities(predicted_prob, numAnswer = 3):
    top_three_answer = predicted_prob.argsort()[-numAnswer:][::-1]
    answer_prob = predicted_prob[top_three_answer]
    answer_prob = answer_prob.tolist()

    predicted_answer = []
    for i in range(numAnswer):
        predicted_answer.append(top_three_answer[i] )

    return( [predicted_answer,answer_prob] )


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def plotImageAndAttentionMap( image_save_dir, attn_map, image_data_path, file_name, purpose):

    # bp()

    img = data_loader.getImage(image_data_path, file_name, purpose)
    height = img.shape[0]
    width = img.shape[1]

    plt.imshow(img)
    plt.title('Original Image')
    plt.savefig(os.path.join( image_save_dir ,'original_image.jpg') )
    # plt.show()
    plt.close('all')

    attn_index = [0,8,17,19,21]

    for i in range( len(attn_map) ):
        plt.imshow( img )
        curr_attn = np.squeeze(attn_map[i])
        plt.imshow( curr_attn, cmap='jet',alpha=0.5,interpolation='nearest')
        # plt.show()
        # attn_file_name = 'attn_t_' + str(attn_index[i]) + '.jpg'
        plt.savefig( os.path.join( image_save_dir, 'attn_t_'+str(attn_index[i])+'.jpg') )
        plt.close('all')





def process_results( top_predicted_answer, predicted_answer_prob, attn_map, image_data_path, question,answer,
                    file_name,image_save_dir, reverse_quest_vocab, reverse_answer_vocab, purpose='train' ):

    log_filename = os.path.join(image_save_dir,'qnA.log')
    fHandle = open( log_filename, 'w')

    fHandle.write('Question:\n')
    for i in range(question.shape[0]):
        if( question[i] != 0):
            fHandle.write( reverse_quest_vocab[question[i]] + '  ')
            print(reverse_quest_vocab[question[i]])

    fHandle.write('\n')
    print('\n')

    fHandle.write("Correct Answer: \n")
    print("Correct Answer: \n")

    fHandle.write(reverse_answer_vocab[np.argmax(answer)])
    print(reverse_answer_vocab[np.argmax(answer)])


    fHandle.write("\nPredicted Answers:\n ")
    print("\nPredicted Answers: \n")

    for i in range( len(top_predicted_answer) ):
        fHandle.write( str(reverse_answer_vocab[top_predicted_answer[i]]) +'\t----------->\t' + str(predicted_answer_prob[i]) + '\n')
        print(reverse_answer_vocab[top_predicted_answer[i]] ,'\t----------->\t', predicted_answer_prob[i] )

    fHandle.close()


    plotImageAndAttentionMap( image_save_dir, attn_map, image_data_path, file_name, purpose)




