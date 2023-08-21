//#include "NeuralNet/Layer.hpp"
#include <chrono>
#include "NeuralNet/Matrix.hpp"

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
    Matrix b(vector<double>({1,2,0,0}),2,2);
    Matrix c(vector<double>({1}),1,1);
    
    cout<<endl;
    a.print();
    cout<<endl;
    auto start = chrono::high_resolution_clock().now();
    auto tmp = a.pool(2, 0, 2);
    auto end = chrono::high_resolution_clock().now();
    cout<<chrono::duration(end - start).count()<<endl;
    tmp.print();
    return 0;
}