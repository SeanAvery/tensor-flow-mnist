import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)
node3 = tf.add(node1, node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
add_n_triple = adder_node * 3

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])

sess.run([fixW, fixb])

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

 
