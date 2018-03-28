# from numpy import exp, array, random, dot
#
# training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# training_set_outputs = array([[0, 1, 1, 0]]).T
# random.seed(1)
# synaptic_weights = 2 * random.random((3, 1)) - 1
# for iteration in range(10000):
#     output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
#     synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
# print(1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))


from numpy import exp, array, random, dot, amax

training_set_inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_set_outputs = array([[0, 0, 0, 1], [0, 1, 1, 1]]).T
random.seed(1)
synaptic_weights = 2 * random.random((2, 2)) - 1
for iteration in range(10000):
    preActivation = dot(training_set_inputs, synaptic_weights)
    output = (preActivation > .5) * 1
    error = (training_set_outputs - output)
    correction = .1 * dot(training_set_inputs.T, error)
    synaptic_weights += correction
    maxError = amax(abs(error))
    if maxError < .1:
        break
o00 = dot([0, 0], synaptic_weights)
o01 = dot([0, 1], synaptic_weights)
o10 = dot([1, 0], synaptic_weights)
o11 = dot([1, 1], synaptic_weights)

print(o00)
print(o01)
print(o10)
print(o11)

print("\n")

print(1 * (o00 > .5))
print(1 * (o01 > .5))
print(1 * (o10 > .5))
print(1 * (o11 > .5))
