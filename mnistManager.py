#this file is to extract the training/testing data to test our generic net on
import gzip, pickle
import time
import numpy as np
import Net

trainingBatchSize = 60000 #num training items to use per iteration
testingBatchSize = 10000 #num testing items to test on
acceptableAccuracy = 0.97 #accuracy achieved when we should stop training
inputSize = (28,28) #the size of the training/testing data array
numHidden = 300 #the number of hidden nodes to train with
numOutput = 10 #10 numbers to classify

#load the training and testing data from the dataset
fileReader = gzip.open('data_training', 'rb')
trainingData = pickle.load(fileReader)
fileReader.close()
fileReader = gzip.open('data_testing', 'rb')
testingData = pickle.load(fileReader)
fileReader.close()

#the time we started training
startTime = time.time() 
#the iterations counter
numIterations = 0
#the net we will use for training
net = Net.Net(inputSize[0]*inputSize[1], numHidden, numOutput)

#start training until acceptable accuracy
while net.getAccuracy() < acceptableAccuracy:
    net.resetScoreRecord() #don't want aggregate over iterations
    print("iteration: ", numIterations) #debug
    print("trained for %f seconds" % (time.time() - startTime)) #debug
    #iterate through each batch item to get the current net input
    for batchNum in range(trainingBatchSize):
        #format the inputs and targets for the net
        netInput = trainingData[0][batchNum] #do we need to give this a second dimension of 1?
        netTarget = np.zeros((1,numOutput))
        netTarget[0, trainingData[1][batchNum]] = 1 #set the correct label index to 1
        #train the net with our formatted inputs/targets
        net.train(netInput, netTarget)

    #reset and print training acuracy for feedback
    print("batch accuracy was ", net.getAccuracy())
    #iterations counter
    numIterations += 1
print("done training, starting testing") #debug

#reset accuracy and time for testing and debug
net.resetScoreRecord()
startTime = time.time()
#start testing the net 
for batchNum in range(testingBatchSize):
    #format the inputs and targets for the net
    netInput = testingData[0][batchNum] #do we need to give this a second dimension of 1?
    netTarget = np.zeros((1,numOutput))
    netTarget[0, testingData[1][batchNum]] = 1 #set the correct label index to 1
    #get the net result with our formatted inputs/targets
    net.test(netInput, netTarget)

print("tested for %f seconds" % (time.time() - startTime)) #debug
print("done testing, accuracy was ", net.getAccuracy()) #debug
