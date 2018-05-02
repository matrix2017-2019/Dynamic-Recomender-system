import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import matplotlib.pyplot as plt

start_time = time.time()

# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)
graph_iter=[]
graph_avg_loss_10=[]
graph_accuracy=[]

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

#1.txt contains data for user1 of activity in increasing sequence.
training_file = 'data/1.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

train_data = read_data(training_file)
print("Loading training file...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(train_data)
vcab_siz = len(dictionary)
learning_rate = 0.001
training_iters = 2000000
display_step = 500
n_input = 3
n_hidden = 64

x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vcab_siz])
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vcab_siz]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vcab_siz]))
}

def GRU(x,w):
    z = tf.split(x,n_input,1)
    r_cell=[]
    r_cell.append(z)
    for i in range(1,n_hidden):
        r_cell.append(w+len(z))
    return r_cell

def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x,n_input,1)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    c=GRU(x,len(outputs))
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    #################################################

    del graph_iter[:]
    del graph_accuracy[:]
    del graph_avg_loss_10[:]

    #################################################


    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(train_data)-end_offset):
            offset = random.randint(0, n_input+1)

        sik = [ [dictionary[ str(train_data[i])]] for i in range(offset, offset+n_input) ]
        sik = np.reshape(np.array(sik), [-1, n_input, 1])

        soo = np.zeros([vcab_siz], dtype=float)
        soo[dictionary[str(train_data[offset+n_input])]] = 1.0
        soo = np.reshape(soo,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: sik, y: soo})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Epoch Size= " + str(step+1) + ", Avg_Loss= " + \
                  "{:.5f}".format(loss_total/display_step) + ", Avg_Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            
            graph_iter.append(step+1)
            graph_accuracy.append(100*acc_total/display_step)
            graph_avg_loss_10.append(loss_total/display_step)

            acc_total = 0
            loss_total = 0
            sym_in = [train_data[i] for i in range(offset, offset + n_input)]
            sym_out = train_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (sym_in,sym_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
    print("Optimization DOne!")
    print("Toatl Elapsed time in Training: ", elapsed(time.time() - start_time))
    
    #####################
    #add plot graph here
    plt.plot(graph_iter,graph_accuracy, linestyle='--', marker='o', color='b',label='Accuracy %')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy %')
    plt.title('Music Dataset')
    plt.show()

    plt.plot(graph_iter,graph_avg_loss_10, linestyle='-', marker='p', color='r',label='Avg Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Avg Loss')
    plt.title('Music Dataset')
    plt.show()
    #####################

    while True:
        prompt = "last %s activity used: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            sik = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(1):
                keys = np.reshape(np.array(sik), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                sik = sik[1:]
                sik.append(onehot_pred_index)
            print("Next Track to recommend: "+sentence.split(' ')[3])
        except:
            print("Track not in dictionary")
        exit(0)
