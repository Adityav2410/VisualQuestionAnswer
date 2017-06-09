import io
import matplotlib.pyplot as plt
import tensorflow as tf


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