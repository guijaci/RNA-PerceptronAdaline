from numpy import array, random, dot, amax

training_set_inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_set_outputs = array([[0, 0, 0, 1], [0, 1, 1, 1]]).T
random.seed(1)
synaptic_weights = 2 * random.random((2, 2)) - 1
for iteration in range(10000):
    # dot = sum of products between training set and the weights
    preActivation = dot(training_set_inputs, synaptic_weights)
    # thresholding
    output = (preActivation > .5) * 1
    # error from training set e = (d - y)
    error = (training_set_outputs - output)
    # w(t+1) = w(t) + n * x * (d - y)
    correction = .1 * dot(training_set_inputs.T, error)
    synaptic_weights += correction

# testing output
o00 = dot([0, 0], synaptic_weights)
o01 = dot([0, 1], synaptic_weights)
o10 = dot([1, 0], synaptic_weights)
o11 = dot([1, 1], synaptic_weights)

# without thresholding (activation fuction)
print(o00)
print(o01)
print(o10)
print(o11)

print("\n")

# with thresholding
print(1 * (o00 > .5))
print(1 * (o01 > .5))
print(1 * (o10 > .5))
print(1 * (o11 > .5))
