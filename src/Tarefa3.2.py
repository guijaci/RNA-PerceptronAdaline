from numpy import array, random, dot, pi, sin, cos, ones, hstack, amax, asscalar

f1 = lambda x: sin(x)
f2 = lambda x: cos(x)
f3 = lambda x: x
zarray = random.random(15) * 2 * pi
training_set_inputs = array(list(map(lambda x: [f1(x), f2(x), f3(x)], zarray)))

# applying F function to f1, f2 and f3 (entries)
training_set_outputs = dot(training_set_inputs, array([[.565, 2.657, .674]]).T) - pi

# appending 1's column to train for the constant (a)
training_set_inputs = hstack((training_set_inputs, ones((15, 1))))

# c - 1 neuron; r - 3 inputs (f1, f2, f3) + 1 constant (a); z pertains to [0, 2*pi)
synaptic_weights = 2 * random.random((4, 1)) * pi
for iteration in range(1000000):
    sumError = 0
    for i in range(15):
        input_row = training_set_inputs[i]
        output_row = asscalar(training_set_outputs[i])
        output = asscalar(dot(input_row, synaptic_weights))
        # error from training set e = (d - y) = (u - y)
        error = output_row - output
        sumError += abs(error)
        # w(t+1) = w(t) + n * x * e
        delta = .05 * array([input_row]).T * error
        synaptic_weights += delta
    # checking finalized condition
    if amax(sumError / 15) < 1E-9:
        break

# testing output
print("\nOutput")
print(dot(training_set_inputs, synaptic_weights))
print("\nTraining data")
print(training_set_outputs)
print("\nSynaptic Weights")
print(synaptic_weights)
