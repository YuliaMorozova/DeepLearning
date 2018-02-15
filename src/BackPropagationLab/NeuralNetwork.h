#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(int input_n, int output_n, double learningRate, int numberOfHiddenNeurons);
	void train(double** Data, double* Labels, int data_size, int numberOfEpochs, double crossError, bool isTrain);
	~NeuralNetwork();

private:
	int input_n;
	int output_n;
	double learningRate;
	int numberOfHiddenNeurons;

	double** hideWeights;
	double** outputWeights;

	double* input;
	double* output;
	double* hide_output;
	double* deltaWeightOnHide;
	double* deltaWeightOnOutput;
	double* grad_hide;
	double* grad_output;

	double* calculate_hide_output(double * input);
	double * calculate_output(double * hide_output);
	void calculate_grad(double *y);
	void change_weights(double * grad_o, double * grad_h);
	void change_delta(double * grad_o, double * grad_h);
	double calculate_crossEntropy(double** Data, double* Labels, int size);
	void mixData(double** Data, double* Labels, int size);
};