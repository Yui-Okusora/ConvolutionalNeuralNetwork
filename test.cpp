//#include "NeuralNet/Layer.hpp"
#include <chrono>
#include "NeuralNet/Matrix.hpp"
#include "NeuralNet/PoolingLayer.hpp"
#include "NeuralNet/Tensor.hpp"

#include <iostream>
#include <vector>
using namespace std;
double *b;

double* &test(){
    return b;
}

int main()
{
    Matrix a(vector<double>({1,0,4,1,0,0,3,1,0,1,6,0,1,0,6,1,0,0,5,0,0,1,1,0,1}),5,5);
    Matrix b(vector<double>({2,3,4,5,6,7,8,9,4}),3,3);
    Matrix c(vector<double>({1}),1,1);
    Tensor d(1, a);
    PoolingLayer layer(5, 5, 3, 3, 1);
    cout<<endl;
    a.print();
    cout<<endl;
    auto start = chrono::high_resolution_clock().now();
    auto tmp = layer.feedForward(d);
    auto end = chrono::high_resolution_clock().now();
    cout<<chrono::duration(end - start).count()<<endl;
    tmp[0].print();
    layer.calcPoolingGradient(Tensor(1, b))[0].print();
    return 0;
}