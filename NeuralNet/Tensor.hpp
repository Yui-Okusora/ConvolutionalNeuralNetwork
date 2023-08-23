#pragma once
#include "Matrix.hpp"
using namespace std;

class Tensor
{
public:
    Tensor(){}
    Tensor(unsigned depth, Matrix _n);
    Tensor(unsigned rows, unsigned cols, unsigned depth);
    Matrix correlation(Tensor &kernels, unsigned padding, unsigned stride);
    Tensor addPadding(unsigned paddingThickness, unsigned top, unsigned bot, unsigned left, unsigned right);
    Matrix &operator[](size_t index){ return m_values[index];}
    size_t size(){return m_values.size();}
private:
    vector<Matrix> m_values;
    unsigned n_rows;
    unsigned n_columns;
    unsigned n_depth;
};

Tensor Tensor::addPadding(unsigned paddingThickness, unsigned top = 1, unsigned bot = 1, unsigned left = 1, unsigned right = 1)
{
    Tensor tmp = *this;
    for(unsigned i = 0; i < n_depth; ++i)
    {
        tmp[i] = tmp[i].addPadding(paddingThickness, top, bot, left, right);
    }
    return tmp;
}

Matrix Tensor::correlation(Tensor &kernels, unsigned padding, unsigned stride)
{
    Tensor inp = addPadding(padding);
    Matrix out((inp.n_rows - kernels.n_rows + 2 * padding) / stride - 1, (inp.n_columns - kernels.n_columns + 2 * padding) / stride - 1);
    vector<MatrixRef> subMatPtr;
    subMatPtr.reserve(n_depth);
    for(unsigned i = 0; i < n_depth; ++i)
    {
        subMatPtr[i] = inp[i].submat(0, 0, kernels.n_rows, kernels.n_columns);
    }

    for(unsigned i = 0; i <= inp.n_rows - kernels.n_rows; i += stride)
    {
        for(unsigned j = 0; j <= inp.n_rows - kernels.n_rows; j += stride)
        {
            double sum = 0;
            for(unsigned k = 0; k < n_depth; ++k)
            {
                MatrixRef tmp = subMatPtr[k].move(i, j);
                sum += kernels[k].multiply(tmp).sum();
            }
            out.getValue(i / stride, j / stride) = sum;
        }
    }
    return out;
}

Tensor::Tensor(unsigned rows, unsigned cols, unsigned depth)
{
    n_columns = cols;
    n_rows = rows;
    n_depth = depth;
    m_values = vector<Matrix>(depth, Matrix(rows, cols));
}

Tensor::Tensor(unsigned depth, Matrix _n)
{
    n_depth = depth;
    m_values = vector<Matrix>(depth, _n);
    n_columns = _n.getCols();
    n_rows = _n.getRows();
}