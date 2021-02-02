// g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.c -o example`python3-config --extension-suffix`

#include <cmath>
#include <random>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

default_random_engine generator(1);
normal_distribution<double> distribution(0.0, 1.0);

// C++ function

double exponential(double i){
  return exp(i);
}

double * add(double * i, double * j, double num, int size){
  double * result = (double *)malloc(sizeof(double)*size);
  for(int k = 0; k < size; k++){
    result[k] = i[k] + j[k]*num;
    result[k] = exponential(result[k])+distribution(generator);
  }
  return result;
}



// Wrapper

py::array_t<double> wrapper(py::array_t<double> i, py::array_t<double> j, double num) {
  py::buffer_info buf1 = i.request();
  py::buffer_info buf2 = j.request();

  py::array_t<double> result = py::array_t<double>(buf2.size);
  py::buffer_info buf3 = result.request();

  double * i2 = (double *) buf1.ptr;
  double * j2 = (double *) buf2.ptr;
  double * result2 = (double *) buf3.ptr;

  result2 = add(i2,j2, num, buf2.size);
  result = py::array_t<double>(buf2.size, result2);

    return result;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &wrapper, "A function which adds two numbers");
}
