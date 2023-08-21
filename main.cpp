#include <iostream>
#include <random>
#include "NeuralNet\Net.hpp"
#include "mnist\mnist_reader.hpp"
using namespace std;

double uchar2dou(uint8_t a) {return (double)(unsigned)a;}

Matrix hotCodedOutput(uint8_t a)
{
	Matrix tmp(1, 10);
	tmp.getFlatted()[uchar2dou(a)] = 1.0;
	return tmp;
}

double getLabel(Matrix a)
{
	double max = 0;
	unsigned index = 0;
	for(unsigned i = 0; i < 10; ++i)
	{
		if(a.getFlatted()[i] > max)
		{
			max = a.getFlatted()[i];
			index = i;
		}
	}
	return index;
}

int main()
{
	Net mynet;
	auto mnist_data = mnist::read_dataset<vector, vector, double, uint8_t>(100, 5);
	for(unsigned epoch = 0; epoch < mnist_data.training_images.size(); ++epoch)
	{
		Matrix input(mnist_data.training_images[epoch],28, 28);
		Matrix targetOutput = hotCodedOutput(mnist_data.training_labels[epoch]);

		cout << "Epoch: " << epoch + 1 << endl;
		mynet.feedForward(Tensor(1, input));
		cout << "Target: " << uchar2dou(mnist_data.training_labels[epoch]) << endl;
		cout << "Output: " << getLabel(mynet.getOutputVals()) << endl;

		mynet.backProp(targetOutput);
		mynet.calcError(targetOutput);
		cout << "Error: " << mynet.getRecentAverageError() << endl;
		cout << endl;
	}
	return 0;
}