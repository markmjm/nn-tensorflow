import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

observations = 100000
xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(low=-10, high=10, size=(observations, 1))
#
# Combine x and z into an input matric to get 1000 by 2 input matrix
generated_inputs = np.column_stack([xs, zs])
print(f'inputs shape: {generated_inputs.shape}')
#
# Create targets that we will aim to
# In this cas, we have chosen targets = f(x,z) = 2*x - 3*z + 5 + noise .  The funtion is just an example
# 2 is weight1 (w1) and -3 is weight2 (w2) and 5 is the bias(b)
# noise is introduced to randomize the data a bit
noise = np.random.uniform(low=-1, high=+1, size=(observations, 1))
print(f'noise shape: {noise.shape}')
#
# Construct the target
# targets = 2*xs - 3*zs + 5 + noise
generated_targets = 13 * xs - 7 * zs - 12 + noise
print(f'targets shape: {generated_targets.shape}')
#
# Save arrays into a single file in uncompressed .npz format.
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)
#
# Solving with TF
input_size = 2
output_size = 1
### Solving with DF
##### Outline the model ....
##### 1) Define placeholder tf.placeholder ...Note: inputs and targets are defined in the NPZ file
inputs = tf.placeholder(tf.float32, [None,
                                     input_size])  # None means the row dimension is not specided ... No need to specify # of observation, TF can figure it out
targets = tf.placeholder(tf.float32, [None, output_size])
weights = tf.Variable(tf.random_uniform([input_size, output_size], minval=-0.1, maxval=0.1))
biases = tf.Variable(tf.random_uniform([output_size], minval=-0.1, maxval=0.1))
outputs = tf.matmul(inputs, weights) + biases
##### 2) Choose the objective function and the optimization methods ... loss function (L2-norm in this case
mean_loss = tf.losses.mean_squared_error(labels=targets, predictions=outputs) / 2.0
optimize = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(mean_loss)
##### 3) Prepare for Execution
sess = tf.InteractiveSession()
##### 4) Initialize Variables
initializer = tf.global_variables_initializer()  # initialize tf.variables (weighs and biases)
sess.run(initializer)
##### 5) loading training data
training_data = np.load('TF_intro.npz')
##### 6) Learning
for e in range(100):
    _, curr_loss = sess.run([optimize, mean_loss],
                            feed_dict={inputs: training_data['inputs'], targets: training_data['targets']})
    print(curr_loss)

out = sess.run([outputs], feed_dict={inputs: training_data['inputs']})
plt.plot(np.squeeze(out),np.squeeze(training_data['targets']))
plt.xlabel ('output')
plt.xlabel ('target')
plt.show()
#####
##### Get weights
wts = sess.run([weights], feed_dict={inputs: training_data['inputs']})
b = sess.run([biases], feed_dict={inputs: training_data['inputs']})
print(wts)
print(b)