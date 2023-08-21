#pragma once
#include "Matrix.hpp"
using namespace std;

class Tensor : Matrix
{
public:
private:
    vector<Matrix> m_values;
    
};