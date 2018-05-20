# SimpleNeuralNetwork
This program generates a neural network based on user inputs for size.  The user specifies the input length, then passes an array of hidden layer lengths to the constructor for the Network class.  This defines the network to be created.  For exapmle:

Network(2, 3, [4, 5, 5]) generates a network with an input width of 2 and an output witdth of 3.  Between the input and output are three intermediate layers, consisting of 4 nodes, 5 nodes, and 5 nodes.  The entire network is fully connected, and weights and biases are randomly generated.

The program also contains a mutation function, which exists at both the individual Layer level and the Network level.  It takes a "degree" parameter and purturbs the weights and biases in each Layer in the Network it's called on by an amount drawn from a uniform distribution from -degree to degree.

This class is a useful tool for using neuro evolution to train a network for which you already have a good idea of what the optimal structure will be.
