import imageio
import numpy as np
import tensorflow as tf
from model import resnet
import time
import cv2
import pickle
tf.compat.v1.disable_v2_behavior()

def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn)/mx) * 255
    return cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)

def run_depd(image_recieved,IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE):
    startm= time.time()

    x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
    x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    enhanced = resnet(x_image)

    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "models/iphone")
        print("Processing image ")
        image = np.reshape(np.float16(image_recieved) / 255, [1, IMAGE_SIZE])
        print("Enhancing image ")
        start= time.time()
        enhanced_2d = sess.run(enhanced, feed_dict={x_: image})
        enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        print(time.time()-start)
        photo_name = 'nanoJetsun'
        print("Saving Image ")
        # startx = time.time()
        final_image=normalize8(enhanced_image)
        # cv2.imwrite("results/" + "_" + photo_name + "_enhanced.png", final_image)
        # print(time.time()-startx)    
        print('total time',time.time()-startm)
        return final_image