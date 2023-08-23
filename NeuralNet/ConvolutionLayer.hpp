#pragma once
#include "HeaderLibs.hpp"
#include "Tensor.hpp"
using namespace std;

class ConvolutionalLayer;

typedef vector<ConvolutionalLayer> ConvLayer;

class ConvolutionalLayer
{
public:
	ConvolutionalLayer();
	ConvolutionalLayer(unsigned InpHeight, unsigned InpWidth, unsigned InpDepth, unsigned OutNum, unsigned kernelSize, ActivationType Activation);
    //void setOutputVal(unsigned depth, double val) { m_outputVal[depth].setMatrixVal(val); }
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
	unsigned n_depth; // j
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
		for(unsigned depth = 0; depth < n_depth; ++depth)
		{
			Matrix &kernel = kernels[kernelNum][depth];
			Matrix input = prevOutputVals[depth];
			Matrix kernelsGradients = input.correlate(biasesGradients, padding, stride);
			Matrix oldDeltaKernels = deltaKernels[kernelNum][depth];
			Matrix newDeltaKernels = kernelsGradients. multiply( eta ). add( oldDeltaKernels. multiply( alpha ) );
			kernel.copyVals(kernel.add(newDeltaKernels));
		}
	}
}

Tensor ConvolutionalLayer::calcInputGradients(vector<Tensor> nextKernels, Tensor nextGradients, unsigned nextStride)
{
	for(unsigned depth = 0; depth < n_depth; ++depth)
	{
		Matrix sum(m_outputVal[0].getRows(), m_outputVal[0].getCols());
		for(unsigned kernelNum = 0; kernelNum < kernelsNum; ++kernelNum)
		{
			Matrix tmp = nextGradients[kernelNum].correlate(nextKernels[kernelNum][depth].rot180(), 1, nextStride);
			sum = sum.add(tmp);
		}
		Matrix activeFuncDeriv(m_outputVal[depth]);
		for(unsigned i = 0; i < activeFuncDeriv.getFlatted().size(); ++i)
		{
			double &tmp = activeFuncDeriv.getFlatted()[i];
			tmp = activationFunctionDerivative(tmp);
		}

		m_gradients[depth] = sum.multiply(activeFuncDeriv);
	}
	return m_gradients;
}

Tensor ConvolutionalLayer::feedForward(Tensor &input)
{
	assert(input.size() == n_depth);
	assert(input[0].getCols() == inpWidth);
	assert(input[0].getRows() == inpHeight);
	m_inputVal = input;
	for(unsigned kernelNum = 0; kernelNum < kernels.size(); ++kernelNum)
	{
		Tensor kernel = kernels[kernelNum];
		//Zi = Bi
		Matrix sum(biases[kernelNum]);
		
		/*//Zi = Bi + sum(Xj corr Kij)
		for(unsigned depth = 0; depth < channels; ++depth)
		{
			Matrix output = m_inputVal[depth];
			Matrix tmp = output.correlate(kernel[depth], padding, stride);
			sum.copyVals( sum.add( tmp ) );
		}*/

		sum = sum.add(m_inputVal.correlation(kernel, padding, stride));

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

ConvolutionalLayer::ConvolutionalLayer(unsigned InpHeight, unsigned InpWidth, unsigned InpDepth, unsigned OutNum, unsigned KernelSize, ActivationType Activation)
{
	assert(Activation != ActivationType::SoftMax);
	activationType = Activation;
	n_depth = InpDepth;
	kernelsNum = OutNum;
	inpHeight = InpHeight;
	inpWidth = InpWidth;
	createKernel(KernelSize);
	Matrix tmp((InpHeight - KernelSize + 2 * padding) / stride + 1, (InpWidth - KernelSize + 2 * padding) / stride + 1);

	m_gradients = Tensor(OutNum, tmp);
	m_outputVal = m_gradients;
	deltaBiases = m_gradients;
	biases = m_gradients;
	for(unsigned outputDepth = 0; outputDepth < OutNum; ++outputDepth)
	{
		biases[outputDepth] = tmp.setRandomVals(randomWeight);
	}
}

ConvolutionalLayer::ConvolutionalLayer()
{

}

void ConvolutionalLayer::createKernel(unsigned size)
{
	for(unsigned kernelNum = 0; kernelNum < kernelsNum; ++kernelNum)
	{
		deltaKernels.push_back(Tensor(n_depth, Matrix(size, size)));
		kernels.push_back(Tensor(n_depth, Matrix(size, size).setRandomVals(randomWeight)));
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