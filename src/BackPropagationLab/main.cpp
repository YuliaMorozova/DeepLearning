#include "DataReader.h"
#include "NeuralNetwork.h"
#include <string>
#include <vector>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {

	char* trainImagesPath = "data/train-images.idx3-ubyte";
	char* trainLabelsPath = "data/train-labels.idx1-ubyte";
	char* testImagesPath = "data/t10k-images.idx3-ubyte";
	char* testLabelsPath = "data/t10k-labels.idx1-ubyte";

	int numberOfEpochs = 20;
	double crossError = 0.005;
	double learningRate  = 0.01;
	int numberOfHiddenNeurons = 200;

	int input = 28 * 28;
	int output = 10;
	int sizeTrainData = 60000;
	int sizeTestData = 10000;

	double** trainData = new double*[sizeTrainData];
	for(int i = 0; i < sizeTrainData; i++)
		trainData[i] = new double[input];
	double* trainLabels = new double[sizeTrainData];

	ReadData(trainImagesPath, trainData);
	ReadLabels(trainLabelsPath, trainLabels);

	double** testData = new double*[sizeTestData];
	for(int i = 0; i < sizeTestData; i++)
		testData[i] = new double[input];
	double* testLabels = new double[sizeTestData];

	ReadData(testImagesPath, testData);
	ReadLabels(testLabelsPath, testLabels);

	cout << "Creating...\n";
	NeuralNetwork NN = NeuralNetwork(input, output, learningRate, numberOfHiddenNeurons);

	cout << "Train: \n";
	NN.train(trainData, trainLabels, sizeTrainData, numberOfEpochs, crossError, true);
	cout << "\n";
	cout << "Test: \n";
	NN.train(testData, testLabels, sizeTestData, numberOfEpochs, crossError, false);

	system("PAUSE");

	for(int i = 0; i < sizeTrainData; i++)
		delete[] trainData[i];
	delete[] trainData;

	for(int i = 0; i < sizeTestData; i++)
		delete[] testData[i];
	delete[] testData;

	delete[] trainLabels;
	delete[] testLabels;

	return 0;
}