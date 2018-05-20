import numpy
import random as r
import math

class Layer:
    # @dataIn: a vector of input data to this layer of the network
    # @weights:the matrix by which this input data should be multiplied
    #          in order to compute the next layer of the network; the
    #          dimensions will be [length output X length dataIn]
    # @biases: the matrix of biases which will be added to the multiplication
    #          of weights*inputs
    def __init__(self, inLen, outLen):
        self.inLength = inLen;
        self.outLength = outLen;
        self.initializeRandomly();

#.... maybe I should have two kinds of layers, neuron connections and neurons
# that way I won't effectively be cloning the neurons in each layer, that feels bad
# ***actually I think it's fine***
    def propogateData(self, inputs):
        #if len(inputs) != self.inLength:
            #raise ValueError('Input vector width of %d, not equal to layer input width of %d' % (len(inputs), self.inLength));
        preBias = numpy.matmul(self.W, inputs);
        #print("Layer: %d --> %d" %(self.inLength, self.outLength));
        #print("Layer Math\n%s\n X \n%s" %(self.W, inputs))
        #print("\n=\n %s" %(preBias))
        outputs = numpy.add(preBias, self.B);
        #print("%s\n+\n%s\n=\n%s" %(preBias, self.B, outputs))
        outputs = self.getSigmoid()(outputs);
        return outputs;

    # returns a vectorized sigmoid function
    def getSigmoid(self):
        s = lambda d: 1.7159*math.tanh(.6666*d);
        return numpy.vectorize(s);

    # @degree the amount by which each weight is allowed to mutate per generation,
    # expressed as a decimal from [0-1)
    def mutate(self, degree):
        if degree < 0 or degree >= 1:
            raise ValueError('Only mutation degrees between [0, 1) allowed.')
        applyMutation = numpy.vectorize(lambda x: x + (1 - 2*r.random())*degree)
        self.W = applyMutation(self.W);
        self.B = applyMutation(self.B);
        #for index, w in self.W:
        #    self.W[index] = w + (w*degree)*(.5 - r.random())*2;
    #    for index, b in self.B:
        #    self.W[index] = b + (b*degree)*(.5 - r.random())*2;

    def initializeRandomly(self):
        self.W = numpy.random.random([self.outLength, self.inLength]);
        self.B = numpy.random.random([self.outLength]);
        # move the [0-1) range to [-1, 1) instead
        shiftVals = numpy.vectorize((lambda x: (x-.5)*2));
        self.W = shiftVals(self.W);
        self.B = shiftVals(self.B);
        #print("W: "); print(W);
        #print("B: "); print(B);

    def wipeLayer(self):
        self.W = numpy.zeros((self.outLength, self.inLength));
        self.B = numpy.zeros((self.outLength, 1));

    def setLayer(self, weights, biases):
        self.W = weights;
        self.B = biases;

    def __str__(self):
        text = "Layer: %d --> %d\nW (%d x %d): %s \nB (%d x 1): %s" % \
        (self.inLength, self.outLength, len(self.W), len(self.W[0]), self.W, len(self.B), self.B);
        return text;

# networkLayers
class Network:
    def buildNetwork(self):
        self.networkLayers = list();
        for i in range(0, len(self.layerSizeList)):
            # the first layer has inLength of the networks input length
            if i == 0:
                inLength = self.inLength;
            else:
                inLength = self.layerSizeList[i-1];
            outLength = self.layerSizeList[i];
            layer = Layer(inLength, outLength)
            self.networkLayers.append(layer);
        # add the final layer
        self.networkLayers.append(Layer(self.layerSizeList[-1], self.outLength));

    def __init__(self, inLen, outLen, layerSizeList):
        self.inLength = inLen;
        self.outLength = outLen;
        self.layerSizeList = layerSizeList;
        self.buildNetwork();

    def forwardPropogate(self, inputs):
        inputData = inputs;
        for layer in self.networkLayers:
            outputData = layer.propogateData(inputData);
            inputData = outputData;
            print("\nOutput: %s\n" % (outputData))
        return outputData;

    # currently applies a random mutation to all weights and biases
    # maybe we should have a probability to mutate at all, rather
    # than just sometimes you get small numbers
    # we could also use a normalized mutation distribution centered
    # around zero instead of a uniform distribution
    def mutate(self, degree):
        for layer in networkLayers:
            layer.mutate(degree);

    def __str__(self):
        text = "\n***Neural Network***\n\n"
        text+= "Input Width: %d\nOutput Width: %d\n%d Intermediate Layers of %s nodes.\n\n" \
        % (self.inLength, self.outLength, len(self.layerSizeList), self.layerSizeList);
        for layer in self.networkLayers:
            text+=layer.__str__();
            text+="\n";
        return text;



n = Network(3, 1, [4, 5, 7, 3]);
#print(n);
out = n.forwardPropogate([2, 2, 3]);
#out = numpy.matmul([[2, 2], [3,3]], [[1, 0], [0, 1]]);
print(out);
l = Layer(2, 3);
print(l);
l.mutate(.01);
print(l);
