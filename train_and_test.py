#! -*- coding:utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import math

import input_data
import models
import utils

pre_trained_weights = './pretrain/vgg16.npy'
data_dir = '/home/acrobat/DataSets/cifar-10-batches-bin/'
train_log_dir = './logs/train_vgg16/'

IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 10000
IS_PRETRAIN = True


def train():
    with tf.name_scope('input'):
        train_image_batch, train_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=True,
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=True)
        test_image_batch, test_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=False,
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=False)
    logits = models.VGG16(train_image_batch, N_CLASSES, IS_PRETRAIN)

    loss = utils.loss(logits, train_label_batch)
    accuracy = utils.accuracy(logits, train_label_batch)
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = utils.optimize(loss, learning_rate, my_global_step)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # load the parameter file, assign the parameters, skip the specific layers
    utils.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            train_images, train_labels = sess.run([train_image_batch, train_label_batch])
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: train_images, y_: train_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, train_loss: %.4f, train_accuracy: %.4f%%' % (step, train_loss, train_acc))
                summary_str = sess.run(summary_op)
                train_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                test_images, test_labels = sess.run([test_image_batch, test_label_batch])
                test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: test_images, y_: test_labels})
                print('**  Step: %d, test_loss: %.2f, test_accuracy: %.2f%%  **' % (step, test_loss, test_acc))

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

#   Test the accuracy on test dataset. got about 85.69% accuracy.

def test():
    with tf.Graph().as_default():
        n_test = 10000

        images, labels = input_data.read_cifar10(data_dir=data_dir,
                                                 is_train=False,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False)

        logits = models.VGG16(images, N_CLASSES, IS_PRETRAIN)

        correct = utils.num_correct_prediction(logits, labels)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(train_log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found!')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('Testing......')
                num_step = int(math.floor(n_test / BATCH_SIZE))
                num_sample = num_step * BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1
                print('Total testing samples: %d' % num_sample)
                print('Total correct predictions: %d' % total_correct)
                print('Average accuracy: %.2f%%' % (100 * total_correct / num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    train()
    test()
