from numpy import array, random, dot, pi, sin, cos, math

f1 = lambda x: sin(x)
f2 = lambda x: cos(x)
f3 = lambda x: x
training_set_inputs = array(list(map(lambda x: [f1(x), f2(x), f3(x)], random.random(15) * 2 * pi)))

f = lambda x: (-pi + f1(x[0]) * .565 + f2(x[1]) * 2.657 + f3(x[2]) * .674)
training_set_outputs = array([list(map(f, training_set_inputs))]).T

synaptic_weights = 2 * random.random((3, 1)) - 1
for iteration in range(10000):
    # dot = sum of products between training set and the weights
    preActivation = dot(training_set_inputs, synaptic_weights)
    output = preActivation
    # error from training set e = (d - y)
    error = (training_set_outputs - output)
    # w(t+1) = w(t) + n * x * (d - y)
    for i in range(15):
        e = error[i][0]
        x1 = training_set_inputs[i][0]
        x2 = training_set_inputs[i][1]
        x3 = training_set_inputs[i][2]
        x = array([[x1, x2, x3]]).T
        correction = 0.005 * e * x
        synaptic_weights += correction

# testing output
print(dot(training_set_inputs, synaptic_weights))
print('\n')
print(training_set_outputs)
