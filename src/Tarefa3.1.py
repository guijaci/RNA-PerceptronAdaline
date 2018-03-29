from numpy import array, random, dot, amax

training_set_inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_set_outputs = array([[0, 0, 0, 1], [0, 1, 1, 1]]).T
random.seed(1)
# 2 neurons w/ 2 inputs each (c - neurons, r - inputs)
synaptic_weights = 2 * random.random((2, 2)) - 1
for iteration in range(1000000):
    # dot = sum of products between training set and the weights
    preActivation = dot(training_set_inputs, synaptic_weights)
    # threshold (theta = .5)
    output = (preActivation > .5) * 1
    # error from training set e = (d - y)
    error = (training_set_outputs - output)
    # checking training finalized condition
    if amax(abs(error)) < 1E-9:
        break
    # w(t+1) = w(t) + n * x * e
    delta = 1*dot(training_set_inputs.T, error)/4
    synaptic_weights += delta


# testing output
print("Output (w/o activation)")
print(dot(training_set_inputs, synaptic_weights))
print('\n')
# with thresholding
print("Output (w/ activation)")
print(1 * (dot(training_set_inputs, synaptic_weights) > 0.5))
print('\n')
print("Synaptic Weights")
print(synaptic_weights)
