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
    const vector<Tensor> getKernels() {return vector<Tensor>(n_depth, Tensor(3, Matrix(2, 2).setMatrixVal(1.0)));}
    Tensor getOutputVals() {return m_outputVal;}
private:
    Tensor m_outputVal;
    Tensor m_inputVal;
    Tensor m_gradients;

    unsigned stride = 2;
    unsigned padding = 0;
    unsigned size = 2;
    unsigned n_depth;
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
    n_depth = InputChannels;
    m_gradients = Tensor(n_depth, Matrix(inputHeight, inputWidth));
    m_outputVal = Tensor(n_depth, Matrix(outputHeight, outputWidth));
}

Tensor PoolingLayer::Matrix2Tensor(Matrix input)
{
    unsigned index = 0;
    Tensor out(n_depth, Matrix(m_outputVal[0].getRows(), m_outputVal[0].getCols()));
    for(unsigned depth = 0; depth < n_depth; ++depth)
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
        out[depth] = tmp;
    }
    return out;
}

Tensor PoolingLayer::calcPoolingGradient(Tensor nextGradients)
{
    for(unsigned channel = 0; channel < n_depth; ++channel)
    {   
        Matrix input = m_inputVal[channel].addPadding(padding);
        MatrixRef pool = input.submat(0, 0, size, size);
        for(unsigned i = 0; i <= input.getRows() - size; i += stride)
        {
            for(unsigned j = 0; j <= input.getCols() - size; j += stride)
            {
                pool = pool.move(i, j);
                for(unsigned k = 0; k < size; ++k)
                {
                    for(unsigned g = 0; g < size; ++g)
                    {
                        if(pool.getValue(k, g) == m_outputVal[channel].getValue(i / stride, j / stride))
                        {
                            m_gradients[channel].getValue(i + k, j + g) = nextGradients[channel].getValue(i / stride, j / stride);
                        }
                    }
                }
            }
        }
    }
    return m_gradients;
}

Tensor PoolingLayer::feedForward(Tensor input)
{
    m_inputVal = input;
    for(unsigned i = 0; i < n_depth; ++i)
    {
        m_outputVal[i] = input[i].pool(size, padding, stride);
    }
    
    return m_outputVal;
}