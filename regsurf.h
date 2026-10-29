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
#ifndef REGSURF_H 
#define REGSURF_H

#include <iostream>
#include <string>
#include <cmath>
#include <bits/stdc++.h>
#include <fstream>

using namespace std;

# include "mat.h" // vector support libla implementation
# include "vec.h" // compiler parameters: -llapack -lblas
# include "reg.h"

namespace LA {

template <typename T>
unsigned int Build_s_RE_descriptor_mat(const unsigned int count_particles, const unsigned int count_geometries, const T* distances, T* desc, const T* reference_geometry);

template <typename T>
unsigned int Build_s_coulomb_matrix_descriptor_mat(const unsigned int count_particles, const unsigned int count_geometries, const T* distances, T* desc, const unsigned int* Z);

template <typename T>
unsigned int Build_s_inverse_pairs_descriptor_mat(const unsigned int count_particles, const unsigned int count_geometries, const T* distances, T* desc);


// The kernel regression of surfaces of geometries in in 1D, 2D, 3D and multi-D space
// and computing of ML-NEA absorption spectra
template <typename T>
class REGSURF { // REG for potential energy surfaces (PES) and other quantities surfaces
public:
    // constructor of surface REG model from memory locations
    REGSURF(const unsigned int count_dimensions, const unsigned int count_kernels, const unsigned int count_geometries,
    const unsigned int count_levels, const string descriptor, const string kernel, const T* params,  const T* geometries,
    const T* values, const unsigned int* Z = nullptr);
    // constructor of surface REG model from files
    REGSURF(const string geometries_file, const string energies_file, const string parameters_file, const string Z_file = "");
    // inicialization function of REG surface model
    void Init(const unsigned int count_dimensions, const unsigned int count_kernels, const unsigned int count_geometries,
    const unsigned int count_levels, const string descriptor, const string kernel, const T* params,  const T* geometries,
    const T* values, const unsigned int* Z);
    // General kernel function
    T Kernel(const T* x, const unsigned int dimensions, const unsigned int kernel_id);
    // function for building matrix of geometry distances between kernels
    unsigned int Build_s_distances_mat(const unsigned int count_geometries, const T* geometries, T* distances);
    // function for building descriptor matrix
    unsigned int Build_s_descriptor_mat(const unsigned int count_geometries, const T* distances, T* desc);
    // Function, that build matrix of coefficients for kernel coordinates
    unsigned int Build_s_kernels_mat(const unsigned int count_geometries, const T* desc, NRVec<NRMat<T>> &kernels_vec_mat);
    // Function, that build sums of matrix coefficients for kernel coordinates
    unsigned int Build_s_kernels_mat_sums();
    // Function, that return root mean square error of model
    T s_RMSE(unsigned int level); // sqrt(sum(delta_y^2)/n)
    // The Loss function
    T Loss_Function(unsigned int level); // sqrt(sum(delta_y^2)/n) + lambda((1 - alpha)/2 * sum(weight^2) + alpha * sum(abs(weight)))
    
    unsigned int  s_Fit(const T max_Loss, const unsigned int max_learning_cycles, const T learning_rate,
    const bool element_invariant=true, unsigned int* kernel_classes=nullptr);
    // Saving weight coefficients of regression surface model into file
    unsigned int  s_Save_fitted(const string weights);
    // Loading weight coefficients of regression surface model from file
    unsigned int  s_Load_fitted(const string weights);
    // This function is for getting predictions from kernel potential energy surface, geometry is loaded from memory
    unsigned int  s_Get_predictions(const unsigned int count_geometries, const unsigned int count_levels,
    const T* geom, const unsigned int* surfaces, T* predictions);
    // Function, that uses the regression surface to predict y for new data and investigate the model precision
    unsigned int  s_Score(const unsigned int count_geometries, const unsigned int count_levels,
    const T* geom, const unsigned int* surfaces, T* predictions, const T* y_values, T* score);
    // Setting parameters of regression surface models
    unsigned int  s_Set_params(const string descriptor, const string kernel, const T* params);
    // Getting parameters of regression surface models
    unsigned int  s_Get_params(string* descriptor, string* kernel, T* params);
    // Setting the external descriptor function for computing the descriptor matrix
    unsigned int s_Set_extern_descriptor(unsigned int (*extern_func)(const unsigned int, const unsigned int, const T*, T*));
    // Getting weight coefficients of regression surface model
    unsigned int  s_Get_weight_coeff(const unsigned int s_number, T* weights);
        
    // Loading geometries for ML-NEA calculations from memory
    unsigned int s_Load_ML_NEA_geometries(const unsigned int count_geometries, const T* geometries);
    // Loading geometries for ML-NEA calculations from file
    unsigned int s_Load_ML_NEA_geometries(const unsigned int count_geometries, const string geometries_file);
    // Calculation of energies for ML-NEA surfaces of geometries for all energy levels - ground and excited states
    unsigned int s_Predict_ML_NEA_geometries();
    // Calculation of ML-NEA energy spectra
    unsigned int s_Compute_ML_NEA_spectra(const T E_min, const T step, const unsigned int count_steps, const T delta);
    // Saving the normalized ML-NEA energy spectra to file
    unsigned int s_Save_ML_NEA_spectra(const string ML_NEA_spectra_file);
    // destructor of surface REG model 
    ~REGSURF();

protected:
    // string with setted type of kernels
    string  s_kernel_type;
    // identification number of setted type of kernel
    unsigned int s_kernel_id;
    // string with selected type of descriptor for surface
    string  s_descriptor_type;
    // identification number of selected type of descriptor
    unsigned int  s_descriptor_id;
    
    // dimension of kernels in surface geometry
    unsigned int  s_count_dimensions;
    // count of kernels in surface geometry
    unsigned int  s_count_kernels;
    // count of geometries in surface
    unsigned int  s_count_geometries;
    // count of levels/surfaces
    unsigned int  s_count_levels;
    // lambda parameter of regression
    T* s_lambda = nullptr;
    // alpha parameter of regression
    T* s_alpha = nullptr;
    // sigma parameter of kernels
    T s_sigma;
    // n for Matern kernels
    unsigned int s_n;
    // count of geometries for ML-NEA model
    T ML_NEA_count_geometries;
    // minimum of energy in begin of s_ML_NEA_spectra_vec (atomic units)
    T ML_NEA_energy_min;
    // the energy step in s_ML_NEA_spectra_vec (atomic units)
    T ML_NEA_energy_step;
    
    // indicator of initialization of allocation of memory
    bool s_aloc_init = false;
    // indicator of initialization or initialization errors
    bool s_base_init = false;
    // indicator of initialization of s_distances_mat 
    bool s_distances_mat_init = false;
    // indicator of initialization of s_descriptors_mat 
    bool s_descriptors_mat_init = false;
    // indicator of initialization of s_kernels_mat
    bool s_kernels_mat_init = false;
    // indicator of using of extern descriptor functions
    bool extern_descriptor = false;
    // pointer to external descriptor function
    unsigned int (*extern_desc)(const unsigned int, const unsigned int, const T*, T*) = nullptr;
    
    // indicator of loading of ML_NEA input data
    bool s_ML_NEA_input_init = false;
    // indicator of computing of ML_NEA surfaces
    bool s_ML_NEA_surf_init = false;
    // indicator of computing of ML_NEA spectra
    bool s_ML_NEA_spectra_init = false;
    
    // the surface kernel parameters pointer, sigma parameter is first
    T*  s_params = nullptr;
    // Matern kernel coefficient computed from n and k
    T* s_Matern_comb_number = nullptr;
    // numbers of geometries with minimum of y for levels [s_count_levels]
    unsigned int* s_y_min = nullptr;
    // The geometries data of each geometries [s_kernel_dimension * s_count_kernels * s_count_geometries]
    T* s_geometries = nullptr;
    // The surfaces y data values of each surfaces [s_count_geometries * s_count_surf]
    T* s_y = nullptr;
    // The surfaces Z proton numbers of atoms [s_count_kernels]
    unsigned int* s_Z = nullptr;
    // the sum of sums of kernel matrix for base geometry
    T  s_kernels_sum_sum;
    
    // matrix of coordinate distances between kernels for each geometry [s_count_kernels * s_count_kernels * s_count_geometries]
    T* s_distances = nullptr;
    // descriptor matrices for each geometry [s_kernel_dimension * s_kernel_dimension * s_count_geometries]
    T* s_desc = nullptr;
    // matrices of kernel coefficients for each surface kernel geometry [s_count_kernels * s_count_kernels]
    T* s_kernels = nullptr;
    // sums of kernels rows for geometries [s_count_kernels]
    T* s_kernels_rows_sums;
    // matrix of differences of sum kernel rows between geometries [s_count_kernels]
    T* s_kernels_diff;
    // vector differences of sums kernels rows and columns between base and other geometries [s_count_geometries]
    T* s_kernels_diff_sums;
    // vectors of weight coefficients for each surface kernel and surface [s_count_kernels]
    T* s_weights = nullptr;
    
    // matrices of surface y energy/quantity predictions for each geometry and surface [s_count_geometries]
    T*  s_y_preds = nullptr;
    // the surfaces difference y data values of each surfaces [s_count_geometries]
    T*  s_y_delta = nullptr;
    // the surfaces temporary y data values of each surfaces [s_count_kernels]
    T*  s_y_temp = nullptr;
    // the surfaces temporary y partial data values of each surfaces [count_kernels]
    T*  s_y_temp_part = nullptr;
    // coordinate distances for geometry for predictions [s_count_kernels, s_count_kernels]
    T* s_distances_pred;
    // matrix of descriptors between kernels in geometry for predictions [s_count_kernels, s_count_kernels]
    T* s_desc_pred;
    // vector of matrix of kernel coefficients for kernels in geometry for predictions [count_kernels * count_kernels]
    T* s_kernels_pred;
    
    // pointer to x buffer for model loaded from file
    T* s_geom_file = nullptr;
    // pointer to y buffer for model loaded from file
    T* s_y_file = nullptr;
    // pointer to parameters buffer for model loaded from file
    T* s_param_file = nullptr;
    // pointer to fitted data for model loaded from file
    T* s_fit_file = nullptr;
    // pointer to fitted data for model loaded from Z file
    unsigned int* s_Z_file = nullptr;
    // pointer to ML-NEA geometries buffer for model loaded from file
    T* s_ML_NEA_geom_file = nullptr;
    
    // pointer to geometries loaded to ML_NEA model [s_count_dimensions * s_count_kernels, ML_NEA_count_geometries]
    T* s_ML_NEA_geometries = nullptr;
    // ML-NEA surfaces numbers
    unsigned int* s_ML_NEA_surf_numbers;
    // pointer to y values computed for ML_NEA model [ML_NEA_count_geometries, s_count_levels]
    T* s_ML_NEA_y = nullptr;
    // pointer to minimal y values in geometries computed for ML_NEA model [s_count_levels]
    T* s_ML_NEA_min_y = nullptr;
    // pointer to spectra computed for ML_NEA model [count_steps + 1]
    T* s_ML_NEA_spectra = nullptr;
    
    // NRMat for input loaded geometries [s_count_dimensions * s_count_kernels, count_geometries]
    NRMat<T> s_geometries_mat;
    // NRMat for vector of y of geomeries for surface [s_count_geometries, s_count_levels]
    NRMat<T>  s_y_mat;
    // NRVec for vector of Z for coulomb matrix descriptor [s_count_kernels]
    NRVec<unsigned int>  s_Z_vec;
    
    // NRMat for coordinate distances for geometries [s_count_kernels, s_count_kernels * s_count_geometries]
    NRMat<T> s_distances_mat;
    // NRMat for matrix of descriptors between kernels in geometry [s_count_kernels, s_count_kernels * s_count_geometries]
    NRMat<T> s_desc_mat;
    // NRVec of NRMat for matrix of kernel coefficients for kernels in geometry
    // [count_geometries, count_kernels * count_kernels]
    NRVec<NRMat<T>> s_kernels_vec_mat;
    // NRVec of NRVec for sums of kernels rows for geometries [s_count_kernels * s_count_geometries]
    NRVec<NRVec<T>> s_kernels_rows_sums_vec_vec;
    // NRVec of NRVec for differences of sums kernels rows between base and other geometries [s_count_kernels, s_count_geometries]
    NRVec<NRVec<T>> s_kernels_diff_vec_vec;
    // NRVec for differences of sums kernels rows and columns between base and other geometries [s_count_geometries]
    NRVec<T> s_kernels_diff_sums_vec;
    // NRVec of NRVec for vectors of weight coefficients of surface kernels in surface  [s_count_kernels, s_count_levels]
    NRVec<NRVec<T>> s_weights_vec_vec;
    // NRVec for vector of y of geomeries for surface [s_count_geometries]
    NRVec<T> s_y_vec;
    
    // NRVec for vector of y predictions of geomeries for surface [s_count_geometries]
    NRVec<T> s_y_preds_vec;
    // NRVec for vector of delta y values for surface [s_count_geometries]
    NRVec<T> s_y_delta_vec;
    // NRVec for vector of temporary y values for surface [s_count_kernels]
    NRVec<T> s_y_temp_vec;
    // NRVec for vector of temporary partial y values for geometry [s_count_kernels]
    NRVec<T> s_y_temp_part_vec;
    // NRMat for coordinate distances for geometry for predictions [s_count_kernels, s_count_kernels]
    NRMat<T> s_distances_pred_mat;
    // NRMat for matrix of descriptors between kernels in geometry for predictions [s_count_kernels, s_count_kernels]
    NRMat<T> s_desc_pred_mat;
    // NRVec of NRMat for matrix of kernel coefficients for kernels in geometry for predictions [count_kernels * count_kernels]
    NRVec<NRMat<T>> s_kernels_pred_vec_mat;
    
    // NRMat for ML-NEA geometries [s_count_dimensions * s_count_kernels, ML_NEA_count_geometries]
    NRMat<T> s_ML_NEA_geometries_mat;
    // NRVec for ML-NEA surfaces numbers
    NRVec<unsigned int> s_ML_NEA_surf_numbers_vec;
    // NRMat for ML-NEA energies [s_count_levels, ML_NEA_count_geometries]
    NRMat<T> s_ML_NEA_y_mat;
    // NRVec of minimal y values in geometries computed for ML_NEA model [s_count_levels]
    NRVec<T> s_ML_NEA_min_y_vec;
    // NRVec for ML-NEA energy spectra [count_steps + 1]
    NRVec<T> s_ML_NEA_spectra_vec;
};
} // end of namespace
# endif /* REG_H */
