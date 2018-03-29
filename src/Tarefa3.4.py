from numpy import array, random, dot, pi, sin, cos, ones, hstack, amax

f1 = lambda x: sin(x)
f2 = lambda x: cos(x)
f3 = lambda x: x
zArray = random.random(15) * 2 * pi
training_set_inputs = array(list(map(lambda x: [f1(x), f2(x), f3(x)], zArray)))

# applying F function to f1, f2 and f3 (entries)
training_set_outputs = dot(training_set_inputs, array([[.565, 2.657, .674]]).T) - pi

# appending 1's column to train for the constant
training_set_inputs = hstack((training_set_inputs, ones((15, 1))))

# synaptic_weights = array([[.565, 2.657, .674, -pi]]).T
synaptic_weights = 2 * random.random((4, 1)) * pi
for iteration in range(1000000):
    # dot = sum of products between training set and the weights
    output = dot(training_set_inputs, synaptic_weights)
    # error from training set e = (d - y)
    error = (training_set_outputs - output)
    # checking finalized condition
    if amax(abs(error)) < 1E-9:
        break
    # w(t+1) = w(t) + sum (p = 0 .. P, n * x * e) / p
    delta = .1 * dot(training_set_inputs.T, error) / 15
    synaptic_weights += delta

# Set to test the result network
zArray = random.random(15) * 2 * pi
test_set_inputs = array(list(map(lambda x: [f1(x), f2(x), f3(x)], zArray)))

# applying F function to f1, f2 and f3 (entries) F(z) = -pi + .565*f1(z) + 2.657*f2(z) + .674*f3(z)
test_set_outputs = dot(test_set_inputs, array([[.565, 2.657, .674]]).T) - pi

# appending 1's column to train for the constant
test_set_inputs = hstack((test_set_inputs, ones((15, 1))))

test_output = dot(test_set_inputs, synaptic_weights)
test_error = (test_set_outputs - test_output)
avg_test_error = test_error.mean()

# testing output
print("Test Output")
print(test_output)
print('\n')
print("Ground Truth")
print(test_set_outputs)
print('\n')
print("Errors")
print(test_error)
print('\n')
print("Average error")
print(avg_test_error)
print('\n')
print("Synaptic Weights")
print(synaptic_weights)
