#file for a generic neural net with only one hidden layer
import numpy as np

#the activation function used for the nonlinear output (sigmoid)
activationFunction = lambda x: 1.0/(1.0 + np.exp(-x)) 
#the learning rate for adjusting the weights
learningRate = 0.5

#the class for the neural net
class Net:

    #array of 1s if correct prediction 0 other to record accuracy
    scoreRecord = []

    """constructor for the neural net
    @param numInput: the number of input nodes
    @param numHidden: the number of hidden nodes
    @param numOutput: the number of output nodes"""
    def __init__ (self, numInput, numHidden, numOutput):
        #create the weights for the input to hidden w normalized random values
        self.wih = np.random.normal(0.0, pow(numInput,-0.5), \
                (numInput, numHidden))
        #same for hidden to output
        self.who = np.random.normal(0.0, pow(numHidden,-0.5), \
                (numHidden, numOutput))
        print("initial hidden shape is",self.wih.shape)

    """method to train the net
    @param inputArr: the input array vector
    @param target: the target vector to strive towards"""
    def train (self, inputArr, target):
        #format the input to a vector
        inputArr = np.array(inputArr, ndmin=2)
        #backprop the input which will call the feed forward predict 
        self.backpropogate(inputArr, target)
    
    """method to test the net, records accuracy but doesn not backprop
    @param inputArr: the input array vector
    @param target: the target vector to strive towards"""
    def test (self, inputArr, target):
        #format the input to a vector
        inputArr = np.array(inputArr, ndmin=2)
        #get net output to check if correct label or not
        _, outputNodes = self.feedForward(inputArr)
        self.scoreRecord.append(self.checkPrediction(outputNodes, target))

    """method to perform feed forward to net
    @param inputArr: the input array vector to feed forward
    @return: the hidden nodes output, the output nodes output"""
    def feedForward (self, inputArr):
        #calculate both layers
        hiddenNodesPreActivation = np.dot(inputArr, self.wih)
        hiddenNodes = activationFunction(hiddenNodesPreActivation)
        outputNodesPreActivation = np.dot(hiddenNodes, self.who)
        outputNodes = activationFunction(outputNodesPreActivation)
         
        return hiddenNodes, outputNodes

    """method to backpropogate an error 
    @param inputArr: the input array vector to get feed forward
    @param target: the target vector to strive towards"""
    def backpropogate (self, inputArr, target):
        #get the net prediction for this input to calc errors
        hiddenNodes, outputNodes = self.feedForward(inputArr)
        outputError = outputNodes - target
        hiddenError = np.dot(outputError, self.who.T)
        #get the deltas to apply to the weights
        outputDelta = np.dot(((outputNodes*outputError*(1.0-outputNodes))).T, \
                hiddenNodes)
        hiddenDelta = np.dot((hiddenNodes*hiddenError*(1.0-hiddenNodes)).T, \
                inputArr)
        #apply the errors to the weights
        self.who -= learningRate * outputDelta.T
        self.wih -= learningRate * hiddenDelta.T

        #use this item for accuracy measurement
        self.scoreRecord.append(self.checkPrediction(outputNodes, target))

    """method to check if a net prediction was correct or not
    @param output: the output of the net
    @param target: the correct target the net should have predicted
    @return: 1 if the net was correct, 0 if not"""
    def checkPrediction (self, output, target):
        return 1 if np.argmax(output) == np.argmax(target) else 0

    """method to get the current accuracy of the net
    @return: the accuracy of the net as float """
    def getAccuracy (self):
        #mean doesn't work with empty list
        if len(self.scoreRecord) == 0:
            return 0
        return np.mean(self.scoreRecord)
     

    """method to reset the score record to restart accuracy testing"""
    def resetScoreRecord (self):
        self.scoreRecord = []
