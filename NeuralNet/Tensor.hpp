#pragma once
#include "Matrix.hpp"
using namespace std;

class Tensor
{
public:
    Tensor(unsigned depth, Matrix _n);
    Tensor correlation(Tensor kernels, unsigned padding, unsigned stride);
    Matrix &operator[](size_t index){ return m_values[index];}
private:
    vector<Matrix> m_values;
    unsigned n_rows;
    unsigned n_columns;
    unsigned n_depth;
};

Tensor::Tensor(unsigned depth, Matrix _n)
{
    n_depth = depth;
    m_values = vector<Matrix>(depth, _n);
    n_columns = _n.getCols();
    n_rows = _n.getRows();
}