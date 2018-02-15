#include "NeuralNetwork.h"

double random(double min, double max)
{
	return (double)(rand()*(max - min)/((double)RAND_MAX + 0.1) + min);
};

double* softmax(double* input, int size_input)
{
	double* exp_input = new double[size_input];
	for(int i = 0; i < size_input; i++)
		exp_input[i] = exp(input[i]);

	double sum = 0;
	for(int i = 0; i < size_input; i++)
		sum += exp_input[i];

	double* softmax = new double[size_input];
	for(int i = 0; i < size_input; i++)
		softmax[i] = exp_input[i]/sum;

	return softmax;
};

int indexOfMaxElement(double *array, int size_array)
{
	double max = 0.0;
	int index = 0;
	for(int i = 0; i < size_array; i++)
		if(max < array[i])
		{
			max = array[i];
			index = i;
		}
		return index;
};

NeuralNetwork :: NeuralNetwork(int input_n, int output_n, double learningRate, int numberOfHiddenNeurons)
{
	this->input_n = input_n;
	this->output_n = output_n;
	this->learningRate = learningRate;
	this->numberOfHiddenNeurons = numberOfHiddenNeurons;

	input = new double [input_n];
	output = new double [output_n];
	hide_output = new double [numberOfHiddenNeurons];

	deltaWeightOnOutput = new double [output_n];
	for (int j = 0; j < output_n; j++)
		deltaWeightOnOutput[j] = random(-1.0,1.0);

	deltaWeightOnHide = new double [numberOfHiddenNeurons];
	for (int j = 0; j < numberOfHiddenNeurons; j++)
		deltaWeightOnHide[j] = random(-1.0, 1.0);

	grad_hide = new double [numberOfHiddenNeurons];
	for(int i = 0; i < numberOfHiddenNeurons; i++)
		grad_hide[i] = 0;

	grad_output = new double [output_n];
	for(int i = 0; i < output_n; i++)
		grad_output[i] = 0;

	hideWeights = new double*[input_n];
	for(int i = 0; i < input_n; i++)
		hideWeights[i] = new double[numberOfHiddenNeurons];
	for(int i = 0; i < input_n; i++)
		for(int j = 0; j < numberOfHiddenNeurons; j++)
			hideWeights[i][j] = random(-1.0, 1.0);

	outputWeights = new double*[numberOfHiddenNeurons];
	for(int i = 0; i < numberOfHiddenNeurons; i++)
		outputWeights[i] = new double[output_n];
	for(int i = 0; i < numberOfHiddenNeurons; i++)
		for(int j = 0; j < output_n; j++)
			outputWeights[i][j] = random(-1.0, 1.0);
};

void NeuralNetwork :: train(double** data, double* labels, int data_size, int numberOfEpochs, double crossError, bool isTrain)
{
	double* exp_output = new double[output_n];	
	int epoch_count = 0;
	while(epoch_count < numberOfEpochs)
	{
		double countCorrectAnswers = 0.0;
		mixData(data, labels, data_size);

		cout << "\n";
		cout << "Epoch number: " << epoch_count << "\n";

		for(int i = 0; i < data_size; i++)
		{
			for(int j = 0; j < input_n; j++)
				input[j] = data[i][j];

			for(int j = 0; j < output_n; j++)
			{
				exp_output[j] = 0.0;
				if(j == labels[i])
					exp_output[j] = 1.0;
			}

			output = calculate_output(calculate_hide_output(input));
			if(exp_output[indexOfMaxElement(output, output_n)] == 1.0)
				countCorrectAnswers++;

			if(isTrain)
			{
				calculate_grad(exp_output);

				change_weights(grad_output, grad_hide);
				change_delta(grad_output, grad_hide);
			}
			else
			{
				epoch_count = numberOfEpochs;
			}
		}

		double cross_entrophy = 0.0;
		cross_entrophy = calculate_crossEntropy(data, labels, data_size);
		cout << "Cross-entrophy: " << cross_entrophy << "\n";

		double accuracy = 0.0;
		accuracy = countCorrectAnswers/data_size;

		cout << "Accuracy: " << accuracy << "\n";

		epoch_count++;

		if((cross_entrophy <= crossError) || (1 - accuracy <= crossError))
			break;
	}
};

double* NeuralNetwork :: calculate_hide_output(double * input)
{
	double* sum = new double[numberOfHiddenNeurons];
	for(int i = 0; i < numberOfHiddenNeurons; i++)
		sum[i] = 0;

	for(int i = 0; i< numberOfHiddenNeurons; i++)
		for(int j = 0; j < input_n; j++)
			sum[i] += input[j] * hideWeights[j][i];

	for(int i = 0; i < numberOfHiddenNeurons; i++)
		sum[i] += deltaWeightOnHide[i];

	for(int i = 0; i < numberOfHiddenNeurons; i++)
		hide_output[i] = 1.0/(1.0 + exp(-sum[i]));

	delete[] sum;
	return hide_output;
};

double* NeuralNetwork :: calculate_output(double *hide_output)
{
	double* sum = new double[output_n];
	for(int i = 0; i < output_n; i++)
		sum[i] = 0;

	for(int i = 0; i < output_n; i++)
		for(int j = 0; j < numberOfHiddenNeurons; j++)
			sum[i] += hide_output[j]*outputWeights[j][i];

	for(int i = 0; i < output_n; i++)
		sum[i] += deltaWeightOnOutput[i];

	output = softmax(sum, output_n);

	delete[] sum;
	return output;	
};

void NeuralNetwork :: calculate_grad(double * T)
{
	for(int i = 0; i < output_n; i++)
		grad_output[i] = (T[i] - output[i]);

	double sum = 0.0;
	double d = 0.0;
	for(int i = 0; i < numberOfHiddenNeurons; i++)
	{
		for(int j = 0; j < output_n ; j++)
			sum += grad_output[j]*outputWeights[i][j];

		d = hide_output[i] * (1 - hide_output[i]);
		grad_hide[i] = sum*d;
	}
};

void NeuralNetwork :: change_weights(double * grad_o, double * grad_h)
{
	double delta_weight = 0.0;
	for(int i = 0; i < numberOfHiddenNeurons; i++)
		for(int j = 0; j < output_n; j++)
		{
			delta_weight = learningRate*grad_o[j]*hide_output[i];
			outputWeights[i][j] += delta_weight;
		}

		for(int i = 0; i < input_n; i++)
			for (int j = 0; j < numberOfHiddenNeurons; j++)
			{
				delta_weight = learningRate*grad_h[j]*input[i];
				hideWeights[i][j] += delta_weight;
			}
};

void NeuralNetwork :: change_delta(double * grad_o, double * grad_h)
{
	double delta = 0.0;
	for(int j = 0; j < output_n; j++)
	{
		delta = learningRate*grad_o[j];
		deltaWeightOnOutput[j] += delta;
	}

	for(int j = 0; j < numberOfHiddenNeurons; j++)
	{
		delta = learningRate*grad_h[j];
		deltaWeightOnHide[j] += delta;
	}

};

double NeuralNetwork :: calculate_crossEntropy(double** Data, double* Labels, int size)
{
	double sum = 0.0;

	double *X = new double[input_n];
	double *Y = new double[output_n];
	double *T = new double[output_n];

	for(int i = 0; i < size;i++)
	{
		for(int j = 0; j < input_n; j++)
			X[j] = Data[i][j];

		for(int j = 0; j < output_n; j++)
		{
			T[j] = 0.0;
			if (j == Labels[i])
				T[j] = 1.0;
		}

		Y = calculate_output(calculate_hide_output(X));
		for(int j = 0; j < output_n;j++)
			sum +=  log(Y[j]) * T[j];
	}
	return -sum/size;
};

void NeuralNetwork :: mixData(double** Data, double* Labels, int size) 
{
	for(int i = 0; i < size; i++)
	{
		int position1 = rand() % size;
		int position2 = rand() % size;

		swap(Data[position1], Data[position2]);
		swap(Labels[position1], Labels[position2]);
	}
}

NeuralNetwork :: ~NeuralNetwork(void)
{
	delete[] input;
	delete[] output;
	delete[] hide_output;
	delete[] deltaWeightOnHide;
	delete[] deltaWeightOnOutput;
	delete[] grad_hide;
	delete[] grad_output;

	for(int i = 0; i < input_n; i++)
		delete[] hideWeights[i];
	delete[] hideWeights;

	for(int i = 0; i < numberOfHiddenNeurons; i++)
		delete[] outputWeights[i];
	delete[] outputWeights;

	delete[] hideWeights;
	delete[] outputWeights;
}