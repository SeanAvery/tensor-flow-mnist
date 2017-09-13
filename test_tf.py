import tensorflow as tf

hello = tf.constant('Hello world, yo')
sess = tf.Session()
print(sess.run(hello))
