#pragma once

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include "Matrix.hpp"

enum ActivationType
{
	ReLU,
	LReLU,
	Sigmoid,
	Tanh,
    SoftMax
};