#pragma once
#include "HeaderLibs.hpp"

class FullyConnLayer
{
public:
    FullyConnLayer();
    FullyConnLayer(unsigned inpNeuronNum, unsigned neuronNum, ActivationType activation);
    Matrix& getOutputVals() {return m_outputVal;}
    Matrix &getInputWeights() {return weights;}
    Matrix &getInputGradients() {return m_gradients;}
    Matrix feedForward(Matrix input);
    void calcOutputGradients(Matrix targetVals);
    void calcHiddenGradients(Matrix &nextWeights, Matrix &nextGradients);
    void updateInpWeights(Matrix prevOutputVals);
    unsigned getInputsNum() {return weights.getRows() - 1;}
private:
    void createWeights(unsigned inpSize);
    double sumDerivativeOWeight(unsigned j, Matrix &nextWeights, Matrix &nextGradients);
    static double eta; // [0.0...1.0] overall net training rate
	static double alpha; // [0.0...n] multiplier of last weight change [momentum]
	static double reluParam;
	double activationFunction(double x);
	double activationFunctionDerivative(double x);
    double softmax(double x);
	static double sigmoid(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX / 1.5); }
    double sumE_z = 1.0;
    Matrix m_outputVal;
    Matrix weights;
    Matrix deltaWeights;
    Matrix m_gradients;
    ActivationType activationType;
};

double FullyConnLayer::eta = 0.155; // overall net learning rate
double FullyConnLayer::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]
double FullyConnLayer::reluParam = 0.01;

void FullyConnLayer::updateInpWeights(Matrix prevOutputVals)
{
    prevOutputVals.getFlatted().push_back(1.0);
    prevOutputVals.getCols() += 1;
    for(unsigned i = 0; i < weights.getCols(); ++i)
    {
        for(unsigned j = 0; j < weights.getRows(); ++j)
        {
            double oldDeltaWeight = deltaWeights.getValue(j, i);
            double newDeltaWeight = eta * prevOutputVals.getValue(0, j) * m_gradients.getValue(0, i) + alpha * oldDeltaWeight;
            deltaWeights.getValue(j, i) = newDeltaWeight;
            weights.getValue(j, i) += newDeltaWeight;
        }
    }
}

double FullyConnLayer::sumDerivativeOWeight(unsigned j, Matrix &nextWeights, Matrix &nextGradients)
{
    double sum = 0.0;
    for(unsigned i = 0; i < nextWeights.getCols(); ++i)
    {
        sum += nextWeights.getValue(j, i) * nextGradients.getValue(0, i);
    }
    return sum;
}

void FullyConnLayer::calcHiddenGradients(Matrix &nextWeights, Matrix &nextGradients)
{
    for(unsigned i = 0; i < m_gradients.getCols(); ++i)
    {
        double dow = sumDerivativeOWeight(i, nextWeights, nextGradients);
        m_gradients.getValue(0, i) = dow * activationFunctionDerivative(m_outputVal.getValue(0, i));
    }
}

void FullyConnLayer::calcOutputGradients(Matrix targetVals)
{
    for(unsigned i = 0; i < m_outputVal.getCols(); ++i)
    {
        double delta = -(((1.0 - targetVals.getValue(0, i)) / (1.0 - m_outputVal.getValue(0, i))) + (targetVals.getValue(0, i) / m_outputVal.getValue(0, i)));// / targetVals.getFlatted().size();
        m_gradients.getValue(0, i) = delta * activationFunctionDerivative(m_outputVal.getValue(0, i));
    }
}

Matrix FullyConnLayer::feedForward(Matrix input)
{
    input.getFlatted().push_back(1.0);
    input.getCols() += 1;
    m_outputVal = input.dot(weights);
    if(activationType == ActivationType::SoftMax)
    {
        sumE_z = 0.0;
        for(unsigned i = 0; i < m_outputVal.getFlatted().size(); ++i)
        {
            sumE_z += exp(m_outputVal.getFlatted()[i]);
        }
    }
    for(unsigned i = 0; i < m_outputVal.getRows(); ++i)
    {
        for(unsigned j = 0; j < m_outputVal.getCols(); ++j)
        {
            double &tmp = m_outputVal.getValue(i, j);
            tmp = activationFunction(tmp);
        }
    }
    return m_outputVal;
}

FullyConnLayer::FullyConnLayer(unsigned inpNeuronNum, unsigned neuronNum, ActivationType activation)
{
    activationType = activation;
    m_outputVal = Matrix(1, neuronNum);
    m_gradients = Matrix(1, neuronNum);
    createWeights(inpNeuronNum);
}

FullyConnLayer::FullyConnLayer()
{
    
}

void FullyConnLayer::createWeights(unsigned inpSize)
{
    Matrix tmp(inpSize + 1, m_outputVal.getCols());
    weights = tmp;
    deltaWeights = tmp;
    for(unsigned i = 0; i < weights.getRows(); ++i)
    {
        for(unsigned j = 0; j < weights.getCols(); ++j)
        {
            weights.getValue(i, j) = randomWeight();
        }
    }
}

double FullyConnLayer::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double FullyConnLayer::softmax(double x)
{
    return exp(x) / sumE_z;
}

double FullyConnLayer::activationFunction(double x)
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
    case ActivationType::SoftMax:
        return softmax(x);
	default:
		return 0.0;
		break;
	}
}

double FullyConnLayer::activationFunctionDerivative(double x)
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
    case ActivationType::SoftMax:
        {
            //double sm = softmax(x);
            //return sm * (1.0 - sm);
            return x * (1.0 - x);
        }
	default:
		return 0.0;
		break;
	}
}