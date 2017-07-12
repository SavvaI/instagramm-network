import numpy as np
import tensorflow as tf
import pickle
import os
import scipy.misc
import keras.preprocessing.image

def cnn(inputs, depths):
    outputs = inputs
    for i in range(len(depths[:-1])):
        with tf.variable_scope("conv" + str(i)):
            outputs = tf.layers.conv2d(outputs, depths[i], [5, 5], strides=(2, 2), padding='SAME')
            outputs = tf.nn.relu(tf.layers.batch_normalization(outputs))
            # outputs = tf.nn.dropout(outputs, keep_prob = 0.5)
    # outputs = tf.nn.max_pool(outputs, [1, 5, 5, 1], strides=(1, 4, 4, 1), padding='SAME')
    print(outputs)
    with tf.variable_scope("softmax"):
        outputs = tf.reshape(outputs, shape=[-1, outputs.shape.as_list()[1] * outputs.shape.as_list()[2] * outputs.shape.as_list()[3]])
        outputs = tf.layers.dense(outputs, depths[-1], activation=None)
    return outputs


class Batch_Generator():
    def read_example(self, file):
        try:
            y = pickle.load(file)
        except EOFError:
            file.seek(0)
            y = pickle.load(file)
        return y

    def __init__(self, batch_size, path, augmentation = False):
        self.augmentation = augmentation
        self.file = open(path, 'rb')
        self.batch_size = batch_size
        self.augmentor = keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True
        )

    def next_batch(self):
        batch = [self.read_example(self.file) for i in range(self.batch_size)]
        batch = zip(*batch)
        # print np.stack(batch[0]).shape
        # print np.stack(batch[1]).shape
        batch = (np.stack(batch[0]), np.stack(batch[1]))
        if self.augmentation == True:
            batch = self.augmentor.flow(batch[0], batch[1], batch_size=self.batch_size, shuffle=False).next()
        return batch

class Instagram_Net():
    def __init__(self, depths, datapath, batch_size):
        tags = datapath.split('/')[-1].strip('.pkl').split('-')
        self.num2tag = {key: item for key, item in zip(range(len(tags)), tags)}
        self.depths = depths
        self.batch_size = batch_size
        self.batch_generator = Batch_Generator(batch_size, datapath)
        self.x = None
        self.y = None
        self.y_ = None
        self.saver = None
        self.summary_writer = None
        self.sess = tf.Session()

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        self.y_ = tf.placeholder(tf.int32, shape=[None])
        tf.summary.image('image', self.x)
        logits = cnn(self.x / tf.constant(255.0, tf.float32), self.depths)

        self.loss = tf.reduce_mean(

            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits)
        )
        tf.summary.scalar('loss', self.loss)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(self.loss)
        self.y = tf.cast(tf.arg_max(
            tf.nn.softmax(logits, dim=1),
            dimension=1
        ), tf.int32)
        self.accuracy = tf.reduce_mean(
           tf.cast(tf.equal(self.y, self.y_), tf.float32)
        )
        tf.summary.scalar('accuracy', self.accuracy)

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())
        self.summary_ops = tf.summary.merge_all()

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())
    def save_model(self):
        self.saver.save(self.sess, 'models/my_model.ckpt')

    def load_model(self):
        self.saver.restore(self.sess, 'models/my_model.ckpt')

    def clear_logs(self):
        for i in os.walk('logs').next()[2]:
            os.remove(os.path.join('logs', i))

    def classify_folder(self, path):
        files = [os.path.join(path, i) for i in os.walk(path).next()[2]]
        images = []
        for i in files:
            try:
                images.append(scipy.misc.imresize(scipy.misc.imread(i), [256, 256], interp='lanczos'))
            except IOError:
                os.remove(i)
                continue
        batch = np.stack(images, axis=0)
        classes = [self.num2tag[i] for i in list(self.sess.run(self.y, feed_dict={self.x: batch}))]
        print self.num2tag, len(classes), len(os.walk(path).next()[2])
        for i, j in zip(os.walk(path).next()[2], classes):
            os.rename(os.path.join(path, i), os.path.join(path, j + '____' + i))

    def train(self, steps):
        for i in range(steps):
            batch = self.batch_generator.next_batch()
            feed_dict = {self.x: batch[0], self.y_: batch[1]}

            if i % 100 == 0:
                try:
                    self.save_model()
                except KeyboardInterrupt:
                    self.save_model()
                self.summary_writer.add_summary(self.sess.run(self.summary_ops, feed_dict=feed_dict), i)
#            print batch[0].shape, batch[1].shape
            self.sess.run(self.train_op, feed_dict=feed_dict)

#    def predict(self, x):
    def save(self, path=None):
        self.saver.save('models/cnn.ckpt')

n = Instagram_Net([8, 16, 20, 3], 'data/2017-girl-sexygirls.pkl', batch_size=30)
n.clear_logs()
n.build_model()
# n.load_model()
n.initialize()
n.train(5000)
# n.save_model()
# n.classify_folder('test/marydjaaa')

# n.train(1000)

