#pragma once
#include "HeaderLibs.hpp"

using namespace std;


class MatrixRef
{
public:
    MatrixRef(){}
    MatrixRef(unsigned rows, unsigned cols) {n_rows = rows; n_columns = cols; m_refs = vector<double*>(rows * cols, nullptr); m_refs.reserve(rows * cols);}
    double* &getPointerRef(unsigned row, unsigned col);
    double &getValue(unsigned row, unsigned col){return *getPointerRef(row, col);}
    double getMax();
    void setOrigin(unsigned rows, unsigned cols){orig_rows = rows; orig_cols = cols;}
    MatrixRef move(unsigned row, unsigned col);
    
    /*Matrix add(Matrix b);
    Matrix add(double b);
    Matrix subtr(Matrix b);
    Matrix transpose();
    Matrix dot(Matrix &b);
    Matrix multiply(Matrix b);
    Matrix multiply(double x);*/
private:
    vector<double*> m_refs;
    unsigned n_columns = 0;
    unsigned n_rows = 0;
    unsigned orig_cols = 0;
    unsigned orig_rows = 0;
    unsigned start_col = 0;
    unsigned start_row = 0;
};

MatrixRef MatrixRef::move(unsigned row, unsigned col)
{
    MatrixRef tmp = *this;
    for(unsigned i = 0; i < tmp.m_refs.size(); ++i)
    {
        tmp.m_refs[i] = tmp.m_refs[i] + (row * orig_cols + col);
    }
    return tmp;
}

double* &MatrixRef::getPointerRef(unsigned row, unsigned col)
{
    assert(row < n_rows);
    assert(col < n_columns);
    return m_refs[n_rows * row + col];
}

/*Matrix MatrixRef::add(Matrix b)
{
    assert(n_rows == b.getRows());
    assert(n_columns == b.getCols());
    Matrix tmp(n_rows, n_columns);
    for(unsigned i = 0; i < n_rows; ++i)
    {
        for(unsigned j = 0; j < n_columns; ++j)
        {
            tmp.getValue(i, j) = getValue(i, j) + b.getValue(i, j);
        }
    }
    return tmp;
}*/

double MatrixRef::getMax()
{
    double maxx = numeric_limits<double>::min();
    for(unsigned i = 0; i < m_refs.size(); ++i)
    {
        maxx = max(maxx, *m_refs[i]);
    }
    return maxx;
}