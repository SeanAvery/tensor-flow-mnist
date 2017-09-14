import tensorflow as tf

# MODEL PARAMS
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# MODEL I/O 
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# LOSS
loss = tf.reduce_sum(tf.square(linear_model - y))

# OPTIMIMZER
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# TRAINING DATA
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]


# SESSION
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# TRAINING LOOP
for i in range(1000):
    sess.run(train, {x: x_train , y: y_train})

# ACCURACY
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})

print('W: %s b: %s loss: %s'%(curr_W, curr_b, curr_loss))
