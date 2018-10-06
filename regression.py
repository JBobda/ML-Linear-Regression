import numpy as np
import matplotlib.pyplot as plt

#Linear Regression machine learning algorithm using Gradient Descent

def calculateError(b, m, data):
    #Cost function

    totalError = 0
    #Add up all of the squared differences between the data and line predictions
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        #(Label - prediction)^2
        totalError += (y - (m * x + b)) **2
    
    #Return the average of the cost function
    return totalError/ float(len(data))

def stepGradient(currentB, currentM, data, learningRate):
    #Gradient Descent
    gradientB = 0
    gradientM = 0
    numPoints = float(len(data))

    #Finding the average gradient across the whole function, then updating values
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]

        #Calculate the Gradient for M and B with the partial derivative with respect to M and B
        gradientB += -(2/numPoints) * (y - (currentM * x + currentB))
        gradientM += -(2/numPoints) * x * (y - (currentM * x + currentB))
    
    #Move the M and B values in the opposite direction of the gradient
    newB = currentB - (learningRate * gradientB)
    newM = currentM - (learningRate * gradientM)

    return [newB, newM]

def gradientDescentRunner(data, startB, startM, learningRate, iterations):
    #Sets the slope and y-intercept to the initial values in the main function
    b = startB
    m = startM

    #Runs gradient descent algorithm on slope and y-intercept for the iterations specified
    for i in range(iterations):
        b, m = stepGradient(b, m, np.array(data), learningRate)
    
    #After the program has learned the line of best fit, it returns the slope and intecept
    return [b, m]

def main():
    #Load the Student data into an array using numpy
    data = np.genfromtxt("data.csv", delimiter=",")

    #Load the data points into seperate lists for plotting later
    xValues = []
    yValues = []
    for i in range(0, len(data)):
        xValues.append(data[i, 0])
        yValues.append(data[i, 1])
    
    #Hyperparameters that manipulate the machine learning model
    learningRate = 0.0001
    iterations = 1000
    #Initial values for the Line y = mx + b, slope formula
    initialB = 0
    initialM = 0
    #Calling the gradient descent runner function and storing its returns
    [b, m] = gradientDescentRunner(data, initialB, initialM, learningRate, iterations)
    error = str(calculateError(b, m, data))
    print("The line of best fit is y = " + str(m) + "x + " + str(b) + " with an error of " + error)

    #Using Matlplotlib to draw the data, and the points to the screen
    # Create a list of values in the best fit line
    bestFitLine = [m * i + b for i in xValues]

    #Draw the data points from the .csv file
    plt.plot(xValues, yValues, "o")
    #Draw the best fit line
    plt.plot(xValues, bestFitLine)
    #Graph specifications
    plt.ylabel("Test Score received")
    plt.xlabel("Time Spent Studying (Hours)")
    plt.show()

if __name__ == "__main__":
    main()