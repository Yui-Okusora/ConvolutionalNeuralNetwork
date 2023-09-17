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
	Net mynet("mynet1");
	auto mnist_data = mnist::read_dataset<vector, vector, double, uint8_t>(10000, 5);
	unsigned minibatch = 100;
	for(unsigned lap = 0; lap < mnist_data.training_images.size() / minibatch; ++lap)
	{
		for(unsigned epoch = 0; epoch < 100; ++epoch)
		{
			Matrix input(mnist_data.training_images[lap * minibatch + epoch],28, 28);
			Matrix targetOutput = hotCodedOutput(mnist_data.training_labels[lap * minibatch + epoch]);

			cout << "Epoch: " << lap * minibatch + epoch + 1 << endl;
			mynet.feedForward(mynet.normalize(Tensor(1, input), 0, 255));
			cout << "Target: " << uchar2dou(mnist_data.training_labels[epoch]) << endl;
			cout << "Output: " << getLabel(mynet.getOutputVals()) << endl;

			mynet.backProp(targetOutput, 0.1);
			mynet.calcError(targetOutput);
			cout << "Error: " << mynet.getRecentAverageError() << endl;
			cout << endl;
		}
		mynet.saveNet();
	}
	
	return 0;
}