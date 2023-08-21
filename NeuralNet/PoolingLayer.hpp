#pragma once
#include "HeaderLibs.hpp"
using namespace std;

class PoolingLayer
{
public:
    PoolingLayer();
    PoolingLayer(unsigned InputHeight, unsigned InputWidth, unsigned OutputHeight, unsigned OutputWidth, unsigned InputChannels);
    Tensor feedForward(Tensor input);
    Tensor calcPoolingGradient(Tensor nextGradients);
    Tensor Matrix2Tensor(Matrix input);
    Tensor getInputGradient() {return m_gradients;}
    const vector<Tensor> getKernels() {return vector<Tensor>(channels, Tensor(3, Matrix(2, 2).setMatrixVal(1.0)));}
    Tensor getOutputVals() {return m_outputVal;}
private:
    Tensor m_outputVal;
    Tensor m_inputVal;
    Tensor m_gradients;

    unsigned stride = 2;
    unsigned padding = 0;
    unsigned size = 2;
    unsigned channels;
    unsigned outputHeight;
    unsigned outputWidth;
    unsigned inputHeight;
    unsigned inputWidth;
};

PoolingLayer::PoolingLayer()
{

}

PoolingLayer::PoolingLayer(unsigned InputHeight, unsigned InputWidth, unsigned OutputHeight, unsigned OutputWidth, unsigned InputChannels)
{
    inputHeight = InputHeight;
    inputWidth = InputWidth;
    outputHeight = OutputHeight;
    outputWidth = OutputWidth;
    channels = InputChannels;
    m_gradients = Tensor(channels, Matrix(inputHeight, inputWidth));
    m_outputVal = Tensor(channels, Matrix(outputHeight, outputWidth));
}

Tensor PoolingLayer::Matrix2Tensor(Matrix input)
{
    unsigned index = 0;
    Tensor out;
    for(unsigned depth = 0; depth < channels; ++depth)
    {
        Matrix tmp(m_outputVal[0].getRows(), m_outputVal[0].getCols());
        for(unsigned i = 0; i < m_outputVal[0].getRows(); ++i)
        {
            for(unsigned j = 0; j < m_outputVal[0].getCols(); ++j)
            {
                tmp.getValue(i, j) = input.getFlatted()[index];
                ++index;
            }
        }
        out.push_back(tmp);
    }
    return out;
}

Tensor PoolingLayer::calcPoolingGradient(Tensor nextGradients)
{
    for(unsigned channel = 0; channel < channels; ++channel)
    {   
        Matrix input = m_inputVal[channel].addPadding(padding);
        for(unsigned i = 0, gradRow = 0; i <= input.getRows() - size; i += stride)
        {
            for(unsigned j = 0, gradCol = 0; j <= input.getCols() - size; j += stride)
            {
                Matrix pool = input.submat(i, j, size, size);
                for(unsigned k = 0; k < size; ++k)
                {
                    for(unsigned g = 0; g < size; ++g)
                    {
                        if(pool.getValue(k, g) == m_outputVal[channel].getValue(i / stride, j / stride))
                        {
                            m_gradients[channel].getValue(i + k, j + g) = nextGradients[channel].getValue(gradRow, gradCol);
                        }

                    }
                }
                ++gradCol;
            }
            ++gradRow;
        }
    }
    return m_gradients;
}

Tensor PoolingLayer::feedForward(Tensor input)
{
    m_inputVal = input;
    for(unsigned i = 0; i < channels; ++i)
    {
        m_outputVal[i] = input[i].pool(size, padding, stride);
    }
    
    return m_outputVal;
}