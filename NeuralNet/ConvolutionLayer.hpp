#pragma once
#include "HeaderLibs.hpp"
using namespace std;

class ConvolutionalLayer;

typedef vector<ConvolutionalLayer> ConvLayer;

class ConvolutionalLayer
{
public:
	ConvolutionalLayer();
	ConvolutionalLayer(unsigned InpHeight, unsigned InpWidth, unsigned ChannelsNum, unsigned KernelsNum, unsigned kernelSize, ActivationType Activation);
    //void setOutputVal(unsigned channel, double val) { m_outputVal[channel].setMatrixVal(val); }
	//Tensor getOutputVal(void) const { return m_outputVal; }
	//void calcOutputGradients(Matrix targetVals);
    Tensor calcInputGradients(vector<Tensor> nextKernels, Tensor nextGradients, unsigned nextStride);
	Tensor feedForward(Tensor &input);
	vector<Tensor> getKernels() {return kernels;}
	Tensor &getInputGradients() {return m_gradients;}
	Tensor getOutputVals() {return m_outputVal;}
	void updateInpKernels(Tensor prevOutputVals);
	void createKernel(unsigned size);
private:
    static double eta; // [0.0...1.0] overall net training rate
	static double alpha; // [0.0...n] multiplier of last weight change [momentum]
	static double reluParam;
	double activationFunction(double x);
	double activationFunctionDerivative(double x);
	static double sigmoid(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX / 1.5); }
    Tensor m_outputVal;
	Tensor m_inputVal;
    vector<Tensor> kernels;
	vector<Tensor> deltaKernels;
	Tensor biases;
	Tensor deltaBiases;
	Tensor m_gradients;
    unsigned stride = 1;
    unsigned padding = 1;
	unsigned channels; // j
	unsigned kernelsNum; // i
	unsigned inpHeight;
	unsigned inpWidth;
	ActivationType activationType;
};

double ConvolutionalLayer::eta = 0.155;
double ConvolutionalLayer::alpha = 0.5;
double ConvolutionalLayer::reluParam = 0.1;

void ConvolutionalLayer::updateInpKernels(Tensor prevOutputVals)
{
	for(unsigned kernelNum = 0; kernelNum < kernelsNum; ++kernelNum)
	{
		Matrix &bias = biases[kernelNum];
		Matrix biasesGradients = m_gradients[kernelNum];
		Matrix oldDeltaBiases = deltaBiases[kernelNum];
		Matrix newDeltaBiases = biasesGradients. multiply( eta ). add( oldDeltaBiases. multiply( alpha ) );
		bias.copyVals(bias.add(newDeltaBiases));
		for(unsigned channel = 0; channel < channels; ++channel)
		{
			Matrix &kernel = kernels[kernelNum][channel];
			Matrix input = prevOutputVals[channel];
			Matrix kernelsGradients = input.correlate(biasesGradients, padding, stride);
			Matrix oldDeltaKernels = deltaKernels[kernelNum][channel];
			Matrix newDeltaKernels = kernelsGradients. multiply( eta ). add( oldDeltaKernels. multiply( alpha ) );
			kernel.copyVals(kernel.add(newDeltaKernels));
		}
	}
}

Tensor ConvolutionalLayer::calcInputGradients(vector<Tensor> nextKernels, Tensor nextGradients, unsigned nextStride)
{
	for(unsigned channel = 0; channel < channels; ++channel)
	{
		Matrix sum(m_outputVal[0].getRows(), m_outputVal[0].getCols());
		for(unsigned kernelNum = 0; kernelNum < kernelsNum; ++kernelNum)
		{
			Matrix tmp = nextGradients[kernelNum].correlate(nextKernels[kernelNum][channel].rot180(), 1, nextStride);
			sum = sum.add(tmp);
		}
		Matrix activeFuncDeriv(m_outputVal[channel]);
		for(unsigned i = 0; i < activeFuncDeriv.getFlatted().size(); ++i)
		{
			double &tmp = activeFuncDeriv.getFlatted()[i];
			tmp = activationFunctionDerivative(tmp);
		}

		m_gradients[channel] = sum.multiply(activeFuncDeriv);
	}
	return m_gradients;
}

Tensor ConvolutionalLayer::feedForward(Tensor &input)
{
	assert(input.size() == channels);
	assert(input[0].getCols() == inpWidth);
	assert(input[0].getRows() == inpHeight);
	m_inputVal = input;
	for(unsigned kernelNum = 0; kernelNum < kernels.size(); ++kernelNum)
	{
		Tensor kernel = kernels[kernelNum];
		//Zi = Bi
		Matrix sum(biases[kernelNum]);
		
		//Zi = Bi + sum(Xj corr Kij)
		for(unsigned depth = 0; depth < channels; ++depth)
		{
			Matrix output = m_inputVal[depth];
			Matrix tmp = output.correlate(kernel[depth], padding, stride);
			sum.copyVals( sum.add( tmp ) );
		}

		//Yi = activate(Zi)
		for(unsigned i = 0; i < sum.getFlatted().size(); ++i)
		{
			double &tmp = sum.getFlatted()[i];
			tmp = activationFunction(tmp);
		}

		m_outputVal[kernelNum].copyVals(sum);
	}
	
	return m_outputVal;
}

ConvolutionalLayer::ConvolutionalLayer(unsigned InpHeight, unsigned InpWidth, unsigned ChannelsNum, unsigned KernelsNum, unsigned KernelSize, ActivationType Activation)
{
	assert(Activation != ActivationType::SoftMax);
	activationType = Activation;
	channels = ChannelsNum;
	kernelsNum = KernelsNum;
	inpHeight = InpHeight;
	inpWidth = InpWidth;
	createKernel(KernelSize);
	Matrix tmp((InpHeight - KernelSize + 2 * padding) / stride + 1, (InpWidth - KernelSize + 2 * padding) / stride + 1);
	for(unsigned outputDepth = 0; outputDepth < kernelsNum; ++outputDepth)
	{
		m_gradients.push_back(tmp);
		m_outputVal.push_back(tmp);
		deltaBiases.push_back(tmp);
		biases.push_back(tmp.setRandomVals(randomWeight));
	}
}

ConvolutionalLayer::ConvolutionalLayer()
{

}

void ConvolutionalLayer::createKernel(unsigned size)
{
	for(unsigned kernelNum = 0; kernelNum < kernelsNum; ++kernelNum)
	{
		deltaKernels.push_back(Tensor(channels, Matrix(size, size)));
		kernels.push_back(Tensor(channels, Matrix(size, size).setRandomVals(randomWeight)));
	}
}

double ConvolutionalLayer::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double ConvolutionalLayer::activationFunction(double x)
{
	switch (activationType)
	{
	case ActivationType::Tanh:
		return tanh(x);
		break;
	case ActivationType::Sigmoid:
		return sigmoid(x);
		break;
	case ActivationType::ReLU:
		return max(0.0, x);
		break;
	case ActivationType::LReLU:
		return (x > 0.0) ? x : reluParam * x;
		break;
	default:
		return 0.0;
		break;
	}
}

double ConvolutionalLayer::activationFunctionDerivative(double x)
{
	switch (activationType)
	{
	case ActivationType::Tanh:
		return 1.0 - x * x;
		break;
	case ActivationType::Sigmoid:
		{
			//double sig = sigmoid(x);
			//return sig * (1.0 - sig);
			return x * (1.0 - x);
		}
		break;
	case ActivationType::ReLU:
		return (x > 0.0) ? 1.0 : 0.0;
		break;
	case ActivationType::LReLU:
		return (x > 0.0) ? 1.0 : reluParam;
	default:
		return 0.0;
		break;
	}
}