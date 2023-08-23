#pragma once
#include "HeaderLibs.hpp"
#include "FullyConnLayer.hpp"
#include "ConvolutionLayer.hpp"
#include "PoolingLayer.hpp"
using namespace std;

class Net
{
public:
    Net();
    void feedForward(Tensor inputVals);
    void backProp(Matrix targetVals);
    double getRecentAverageError(void) const { return m_recentAverageError; }
    Matrix getOutputVals() {return m_outputVals;}
    void calcError(Matrix inputVals);
    static Matrix flatten(Tensor a);
private:
    unsigned inputDepth = 1;
    double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
    //vector<FullyConnLayer> m_layers;
    ConvolutionalLayer layer1;
    ConvolutionalLayer layer2;
    PoolingLayer layer3;
    FullyConnLayer layer4;
    FullyConnLayer layer5;
    Matrix m_outputVals;
    Tensor m_inputVals;
};

double Net::m_recentAverageSmoothingFactor = 100.0;

Matrix Net::flatten(Tensor a)
{
    vector<double> sum;
    for(unsigned depth = 0; depth < a.size(); ++depth)
    {
        Matrix tmp = a[depth];
        for(unsigned i = 0; i < tmp.getFlatted().size(); ++i)
        {
            sum.push_back(tmp.getFlatted()[i]);
        }
    }
    return Matrix(sum, 1, sum.size());
}

void Net::calcError(Matrix targetVals)
{
    m_error = 0.0;
    for(unsigned n = 0; n < m_outputVals.getCols(); ++n)
	{
		double delta = -1.0 * (targetVals.getValue(0, n) * log(m_outputVals.getValue(0, n)) + (1.0 - targetVals.getValue(0, n)) * log(1.0 - m_outputVals.getValue(0, n)));
		m_error += delta;// *delta;
	}
	m_error /= m_outputVals.getCols();
	//m_error = sqrt(m_error);

	m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
}
#pragma region 
/*void Net::feedForward(Matrix inputVals)
{
    assert(inputVals.getCols() == m_layers[0].getInputsNum());
    m_outputVals = inputVals;
    m_inputVals = inputVals;
    for(unsigned layerNum = 0; layerNum < m_layers.size(); ++layerNum)
    {
        m_outputVals = m_layers[layerNum].feedForward(m_outputVals);
    }
}

/*void Net::backProp(Matrix targetVals)
{

    m_layers.back().calcOutputGradients(targetVals);

    for(int layerNum = m_layers.size() - 2; layerNum >= 0; --layerNum)
    {
        Matrix nextWeights = m_layers[layerNum + 1].getInputWeights();
        Matrix nextGradients = m_layers[layerNum + 1].getInputGradients();
        m_layers[layerNum].calcHiddenGradients(nextWeights, nextGradients);
    }

    for(int layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Matrix tmp = m_layers[layerNum - 1].getOutputVals();
        tmp.getFlatted().push_back(1.0);
        tmp.getCols() += 1;
        m_layers[layerNum].updateInpWeights(m_layers[layerNum - 1].getOutputVals());
    }
    Matrix tmp = m_inputVals;
    tmp.getFlatted().push_back(1.0);
    tmp.getCols() += 1;
    m_layers[0].updateInpWeights(m_inputVals);
}*/
#pragma endregion

void Net::feedForward(Tensor inputVals)
{
    m_inputVals = inputVals;
    Tensor output = layer1.feedForward(inputVals);
    cout<<"Layer 1 passed"<<endl;
    output = layer2.feedForward(output);
    cout<<"Layer 2 passed"<<endl;
    output = layer3.feedForward(output);
    cout<<"Layer 3 passed"<<endl;
    Matrix output2 = flatten(output);
    output2 = layer4.feedForward(output2);
    cout<<"Layer 4 passed"<<endl;
    output2 = layer5.feedForward(output2);
    cout<<"Layer 5 passed"<<endl;
    m_outputVals = output2;
}

void Net::backProp(Matrix targetVals)
{
    layer5.calcOutputGradients(targetVals);
    layer4.calcHiddenGradients(layer5.getInputWeights(), layer5.getInputGradients());
    layer3.calcPoolingGradient(layer3.Matrix2Tensor(layer4.getInputGradients()));
    //layer2.calcInputGradients(layer3.getKernels(), layer3.getInputGradient(), 2);
    layer2.getInputGradients() = layer3.getInputGradient();
    layer1.calcInputGradients(layer2.getKernels(), layer2.getInputGradients(), 1);

    layer5.updateInpWeights(layer4.getOutputVals());
    layer4.updateInpWeights(flatten(layer3.getOutputVals()));
    layer2.updateInpKernels(layer1.getOutputVals());
    layer1.updateInpKernels(m_inputVals);
}

Net::Net()
{
    layer1 = ConvolutionalLayer(28, 28, 1, 32, 3, ActivationType::Sigmoid);
    layer2 = ConvolutionalLayer(28, 28, 32, 32, 3, ActivationType::Sigmoid);
    layer3 = PoolingLayer(28, 28, 14, 14, 32);
    layer4 = FullyConnLayer(6272, 128, ActivationType::Sigmoid);
    layer5 = FullyConnLayer(128, 10, ActivationType::SoftMax);
}