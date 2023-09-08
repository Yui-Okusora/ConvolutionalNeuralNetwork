#pragma once
#include "Matrix.hpp"

using namespace std;
class Tensor
{
public:
    Tensor(){}
    Tensor(unsigned depth, Matrix _n);
    Tensor(unsigned rows, unsigned cols, unsigned depth);
    Matrix correlate(Tensor &kernels, unsigned padding, unsigned stride);
    Tensor addPadding(unsigned paddingThickness, unsigned top, unsigned bot, unsigned left, unsigned right);
    Matrix &operator[](size_t index){ assert(index < n_depth); return m_values[index];}
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
    tmp.n_rows += (top + bot) * paddingThickness;
    tmp.n_columns += (left + right) * paddingThickness;
    for(unsigned i = 0; i < n_depth; ++i)
    {
        tmp[i] = tmp[i].addPadding(paddingThickness, top, bot, left, right);
    }
    return tmp;
}

Matrix Tensor::correlate(Tensor &kernels, unsigned padding, unsigned stride)
{
    assert(kernels.n_columns == kernels.n_rows);
    assert(n_depth == kernels.n_depth);
    Tensor inp = *this;
    Matrix out((inp.n_rows - kernels.n_rows + 4 * padding) / stride - 1, (inp.n_columns - kernels.n_columns + 4 * padding) / stride - 1);
    for(unsigned i = 0; i < n_depth; ++i)
    {
        Matrix tmp = inp[i].correlate(kernels[i], padding, stride);
        out = out.add(tmp);
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