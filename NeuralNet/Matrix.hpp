#pragma once
#include "MatrixRef.hpp"
using namespace std;
class Matrix
{
public:
    Matrix();
    Matrix(const vector<double> &matrix, unsigned rows, unsigned cols);
    Matrix(unsigned rows, unsigned cols);
    vector<double> &getFlatted() {return m_values;}
    double &getValue(unsigned row, unsigned col);
    unsigned &getRows() {return n_rows;}
    unsigned &getCols() {return n_columns;}
    void print();
    void copyVals(const Matrix &b);
    Matrix setRandomVals(double randVal(void));
    Matrix setMatrixVal(double x);
    Matrix add(Matrix b);
    Matrix add(double &b);
    Matrix subtr(Matrix b);
    Matrix transpose();
    Matrix dot(Matrix b);
    Matrix multiply(Matrix b);
    Matrix multiply(MatrixRef b);
    Matrix multiply(double x);
    Matrix addPadding(unsigned paddingThickness, unsigned top, unsigned bot, unsigned left, unsigned right);
    Matrix correlate(Matrix kernel, unsigned padding, unsigned stride);
    Matrix convolute(Matrix kernel, unsigned padding, unsigned stride);
    Matrix pool(unsigned size, unsigned padding, unsigned stride);
    MatrixRef submat(unsigned startRow, unsigned startCol, unsigned rowSize, unsigned colSize);
    Matrix rot180();
    double sum();
private:
    vector<double> m_values;
    unsigned n_columns = 0;
    unsigned n_rows = 0;
};

Matrix::Matrix()
{

}

Matrix::Matrix(const vector<double> &matrix, unsigned rows, unsigned cols)
{
    m_values = matrix;
    n_columns = cols;
    n_rows = rows;
}

Matrix::Matrix(unsigned rows, unsigned cols)
{
    n_columns = cols;
    n_rows = rows;
    m_values = vector<double>(rows * cols , 0.0);
}

void Matrix::copyVals(const Matrix &b)
{
    assert(n_rows == b.n_rows);
    assert(n_columns == b.n_columns);
    m_values = b.m_values;
}

double &Matrix::getValue(unsigned row, unsigned col)
{
    assert(row < n_rows);
    assert(col < n_columns);
    return m_values[n_columns * row + col];
}

Matrix Matrix::setRandomVals(double randVal(void))
{
    for(unsigned i = 0; i < m_values.size(); ++i)
    {
        m_values[i] = randVal();
    }
    return *this;
}

Matrix Matrix::setMatrixVal(double x)
{
    Matrix tmp = *this;
    for(unsigned i = n_columns * n_rows; i > 0; --i)
    {
        tmp.getFlatted()[i] = x;
    }
    return tmp;
}

Matrix Matrix::add(Matrix b)
{
    assert(n_rows == b.n_rows);
    assert(n_columns == b.n_columns);
    Matrix tmp(n_rows, n_columns);
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < n_columns; ++j)
        {
            tmp.getValue(i, j) = getValue(i, j) + b.getValue(i, j);
        }
    }
    return tmp;
}

Matrix Matrix::add(double &b)
{
    Matrix tmp(n_rows, n_columns);
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < n_columns; ++j)
        {
            tmp.getValue(i, j) = getValue(i, j) + b;
        }
    }
    return tmp;
}

Matrix Matrix::subtr(Matrix b)
{
    assert(n_rows == b.n_rows);
    assert(n_columns == b.n_columns);
    Matrix tmp(n_rows, n_columns);
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < n_columns; ++j)
        {
            tmp.getValue(i, j) = getValue(i, j) - b.getValue(i, j);
        }
    }
    return tmp;
}

Matrix Matrix::transpose()
{
    Matrix tmp(n_columns, n_rows);
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < n_columns; ++j)
        {
            tmp.getValue(j, i) = getValue(i, j);
        }
    }
    return tmp;
}

Matrix Matrix::multiply(double x)
{
    Matrix tmp(n_rows, n_columns);
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < n_columns; ++j)
        tmp.getValue(i, j) = getValue(i, j) * x;
    }
    return tmp;
}

Matrix Matrix::multiply(Matrix b)
{
    assert(n_rows == b.n_rows);
    assert(n_columns == b.n_columns);
    Matrix tmp(n_rows, n_columns);
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < n_columns; ++j)
        tmp.getValue(i, j) = getValue(i, j) * b.getValue(i, j);
    }
    return tmp;
}

Matrix Matrix::multiply(MatrixRef b)
{
    assert(n_rows == b.getRows());
    assert(n_columns == b.getCols());
    Matrix tmp(n_rows, n_columns);
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < n_columns; ++j)
        tmp.getValue(i, j) = getValue(i, j) * b.getValue(i, j);
    }
    return tmp;
}

Matrix Matrix::dot(Matrix b)
{
    assert(n_columns == b.n_rows);
    Matrix tmp(n_rows, b.n_columns);
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < b.n_columns; ++j)
        {
            for (unsigned k = 0; k < n_columns; ++k)
            tmp.getValue(i,j) += getValue(i, k) * b.getValue(k, j);
        }
    }
    return tmp;
}

Matrix Matrix::addPadding(unsigned paddingThickness = 0, unsigned top = 1, unsigned bot = 1, unsigned left = 1, unsigned right = 1)
{
    Matrix tmp(n_rows + (top + bot) * paddingThickness, n_columns + (left + right) * paddingThickness);
    for(unsigned i = top * paddingThickness; i < n_rows + top * paddingThickness; ++i)
    {
        for(unsigned j = left * paddingThickness; j < n_columns + left * paddingThickness; ++j)
        {
            double tmp2 = getValue(i - top * paddingThickness, j - left * paddingThickness);
            tmp.getValue(i, j) = tmp2;
        }
    }
    return tmp;
}

Matrix Matrix::correlate(Matrix kernel, unsigned padding, unsigned stride)
{
    assert(kernel.n_columns == kernel.n_rows);
    Matrix tmp = addPadding(padding);
    Matrix out((tmp.n_rows - kernel.n_rows + 2 * padding) / stride - 1, (tmp.n_columns - kernel.n_columns + 2 * padding) / stride - 1);
    for(unsigned inp_row = 0; inp_row <= (tmp.n_rows - kernel.n_rows); inp_row += stride)
    {
        for(unsigned inp_col = 0; inp_col <= (tmp.n_columns - kernel.n_columns); inp_col += stride)
        {
            double sum = 0;
            for(unsigned k = 0; k < kernel.n_rows; ++k)
            {
                for(unsigned g = 0; g < kernel.n_columns; ++g)
                {
                    sum += kernel.getValue(k, g) * tmp.getValue(inp_row + k, inp_col + g);
                }
            }
            out.getValue(inp_row / stride, inp_col / stride) = sum;
            
        }
        
    }
    return out;
}

Matrix Matrix::convolute(Matrix kernel, unsigned padding, unsigned stride)
{
    kernel = kernel.rot180();
    return correlate(kernel, padding, stride);
}

Matrix Matrix::pool(unsigned size, unsigned padding, unsigned stride)
{
    Matrix tmp = addPadding(1, 0, n_rows % 2, 0, n_columns % 2);
    if(padding != 0) tmp = addPadding(padding);
    Matrix out(ceil((tmp.n_rows - size + 1) / (double)stride), ceil((tmp.n_columns - size + 1) / (double)stride));
    MatrixRef ptr = tmp.submat(0, 0, 2, 2);
    for(unsigned i = 0; i < out.getRows(); ++i)
    {
        for(unsigned j = 0; j < out.getCols(); ++j)
        {
            out.getValue(i, j) = ptr.move(i * stride, j * stride).getMax();

        }
    }
    return out;
}

MatrixRef Matrix::submat(unsigned startRow, unsigned startCol, unsigned rowSize, unsigned colSize)
{
    MatrixRef tmp(rowSize, colSize);
    tmp.setOrigin(n_rows, n_columns);
    for(unsigned i = startRow; i < startRow + rowSize; ++i)
    {
        for(unsigned j = startCol; j < startCol + colSize; ++j)
        {
            tmp.getPointerRef(i - startRow, j - startCol) = &getValue(i, j);
        }
    }
    return tmp;
}

Matrix Matrix::rot180()
{
    Matrix tmp(*this);
    reverse(tmp.m_values.begin(), tmp.m_values.end());
    return tmp;
}

double Matrix::sum()
{
    double sum = 0.0;
    for(unsigned i = 0; i < m_values.size(); ++i)
    {
        sum += m_values[i];
    }
    return sum;
}

void Matrix::print()
{
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < n_columns; ++j)
        cout << setw(3) << m_values[n_rows * i + j] << " ";
        cout << endl;
    }
}