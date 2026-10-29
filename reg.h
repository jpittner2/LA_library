/*
    LA: linear algebra C++ interface library
    Kernel ridge regression module Copyright (C) 2024 
    Pavel Florian <florian43@seznam.cz> and Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <string>
#include <cmath>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;
# include "mat.h" // vector support libla implementation
# include "vec.h" // compiler parameters: -llapack -lblas

# include "la.h"
# ifndef REG_H
# define REG_H

namespace LA {

template <typename T>
T Gaussian_kernel(const T* x, const unsigned int dimensions, const T sigma);

template <typename T>
T Laplacian_kernel(const T* x, const unsigned int dimensions, const T sigma);

template <typename T>
T Exponential_kernel(const T* x, const unsigned int dimensions, const T sigma);

template <typename T>
T Matern_kernel(const T* x, const unsigned int dimensions, const T sigma, const unsigned int n, const T* Matern_comb_number);

template <typename T>
unsigned int Matern_coefficient(const unsigned int n, T* comb_number);



// The kernel regression of points in 1D (curve), 2D (surface), 3D and multi-D space
template <typename T>
class REG { // Kernel regression (REG) library
public:
    // Constructor of kernel regression model from memory location
    REG(const T* x, const T* y, const T* params, const unsigned int data_size, const unsigned int x_dimensions, const string kernel);
    // Constructor of kernel regression model from files
    REG(const string x_file, const string y_file, const string param_file);
    // Init function for constructor
    void Init(const T* x, const T* y, const T* params, const unsigned int data_size, const unsigned int x_dimensions, const string kernel);
    // General kernel function
    T Kernel(const T* x, const unsigned int dimensions, const unsigned int kernel_id);
    // Function, that build matrix of distances between kernels
    unsigned int Build_REG_distances_mat();
    // Function, that build matrix of coefficients for kernel coordinates
    unsigned int Build_REG_kernels_mat();
    // Function, that build sums of matrix coefficients for kernel coordinates
    unsigned int Build_REG_kernels_mat_sums();
    // Function, that return root mean square error of model
    T REG_RMSE(); // sum(delta_y)
    // The Loss function
    T Loss_Function(); // sum(delta_y) + lambda((1 - alpha)/2 * sum(weight^2) + alpha * sum(abs(weight)))
    
    // Function, that learn kernels and set weights
    unsigned int REG_Fit(const T max_Loss, const unsigned int max_learning_cycles);
    // Function, that learn kernels and set weights by gradient descent method
    unsigned int REG_Fit_grad(const T max_Loss, const unsigned int max_learning_cycles, const T learning_rate);
    // Function, that store fitted parameters of regression model to file
    unsigned int REG_Save_fitted(const string model_file);
    // Function, that load fitted parameters of model to file
    unsigned int REG_Load_fitted(const string model_file);
    // Function, that uses the kernel regression model to predict y for new data
    unsigned int REG_Get_predictions(const T* x, T* y_preds, const unsigned int data_size);
    // Function, that uses the kernel regression model to predict y for new data and investigate the model precision
    unsigned int REG_Get_Score(const T* x, T* y_preds, const unsigned int data_size, const T* y, T* score);
    // Setting parameters of kernel regression model
    unsigned int REG_Set_params(const T* params, const string kernel);
    // Getting parameters of kernel regression model
    unsigned int REG_Get_params(T* params, string* kernel);
    // Getting weight coefficients of kernels in regression model
    unsigned int REG_Get_weight_coeff(T* weight);
    // destructor of kernel regression model
    ~REG();

private:
    // string with type of kernel
    string REG_kernel_type;
    // number of type of kernel
    unsigned int REG_kernel_id;
    // Count of x items (dimensions) for kernels
    unsigned int REG_count_dimensions;
    // number of input x values/kernels
    unsigned int REG_count_kernels;
    // lambda parameter of regression
    T REG_lambda;
    // alpha parameter of regression
    T REG_alpha;
    // sigma parameter of kernels
    T REG_sigma;
    // n for Matern kernels
    unsigned int REG_n;
    // indicator of initialization or initialization errors
    bool REG_base_init = false;
    // indicator of initialization of allocation of memory
    bool REG_aloc_init = false;
    // indicator of initialization of REG_REG_distances_mat and REG_REG_kernels_mat
    bool REG_mat_init = false;
    
    // Matern kernel coefficient computed from n and k
    T* REG_Matern_comb_number = nullptr;
    // pointer to parameters of kernel regression, various size, first is lambda, second alpha, third sigma parameter of kernels
    // and fourth the n parameter, if is used Matern kernel 
    T* REG_params = nullptr;
    // variable which contains number of learning cycles of model
    unsigned int learning_cycles = 0;
    
    // x values of kernel regression input data
    T* REG_coordinates = nullptr;
    // y values of kernel regression input data
    T* REG_Y = nullptr;
    
    // matrix of distances between kernels
    T* REG_distances = nullptr;
    // matrix of kernel coefficients for kernel coordinates
    T* REG_kernels = nullptr;
    // vector of row sums of kernel coefficients for kernel coordinates
    T* REG_kernels_sums = nullptr;
    // vector of weight coefficients for each kernel
    T* REG_weights = nullptr;
    
    // vector of distances between kernels
    T* REG_coordinates_diff = nullptr;
    // vector of kernel coefficients for kernel coordinates
    T* REG_kernels_diff = nullptr;
    // vector of y predictions for each kernel
    T* REG_y_preds = nullptr;
    // vector of differences between y values and predictions for each kernel
    T* REG_y_delta = nullptr;
    // vector of temporary results for processing y values and predictions for each kernel
    T* REG_y_temp = nullptr;
    
    // pointer to x buffer for model loaded from file
    T* REG_x_file = nullptr;
    // pointer to y buffer for model loaded from file
    T* REG_y_file = nullptr;
    // pointer to parameters buffer for model loaded from file
    T* REG_param_file = nullptr;
    // pointer to fitted data for model loaded from file
    T* REG_fit_file = nullptr;
    
    // NRVec for Matern kernel coefficient computed from n and k
    NRVec<T> REG_Matern_comb_number_vec;

    // NRVec for matrix of distances between kernels
    NRVec<T> REG_coordinates_vec;
    // NRVec for vector of y imput values of kernels
    NRVec<T> REG_y_vec;
    // NRMat for matrix of distances between kernels
    NRMat<T> REG_distances_mat;
    // NRMat for matrix of kernel coefficients for kernel coordinates
    NRMat<T> REG_kernels_mat;
    // NRVec for vector of row sums of kernel coefficients for kernel coordinates
    NRVec<T> REG_kernels_sums_vec;
    // NRVec for vector of weight coefficients of kernels
    NRVec<T> REG_weights_vec;
    
    // NRVec for matrix of coordinates differences between kernels
    NRVec<T> REG_coordinates_diff_vec;
    // NRVec for matrix of kernel coefficients for kernel coordinates
    NRVec<T> REG_kernels_diff_vec;
    
    // NRVec for vector of y predictions of kernels
    NRVec<T> REG_y_preds_vec;
    // NRVec for vector of differences between estimated real and real y values for training kernels
    NRVec<T> REG_y_delta_vec;
    // NRVec for temporary results for processing y values and predictions for each kernel
    NRVec<T> REG_y_temp_vec;
};
// Include the template classes definitions

} // end of namespace
# endif /* REG_H */
