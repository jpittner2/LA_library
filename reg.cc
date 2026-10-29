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

#include "reg.h"

namespace LA {
template <typename T>
T Gaussian_kernel(const T* x, const unsigned int dimensions, const T sigma)
    {
    unsigned int i;
    T result = 0;
    
    for (i = 0; i < dimensions; i++)
        result = result + (x[i] * x[i]);
        
    result = std::exp(-result/(2 * sigma * sigma));
    return(result);
    }


template <typename T>
T Laplacian_kernel(const T* x, const unsigned int dimensions, const T sigma)
    {
    unsigned int i;
    T result = 0;
    
    for (i = 0; i < dimensions; i++)
        result = result + abs(x[i]);
    
    result = std::exp(-abs(result)/(sigma));
    return(result);
    }


template <typename T>
T Exponential_kernel(const T* x, const unsigned int dimensions, const T sigma)
    {
    unsigned int i;
    T result = 0;
    
    for (i = 0; i < dimensions; i++)
        result = result + (x[i] * x[i]);
    
    result = std::exp(-std::pow(result, 0.5)/sigma);
    return(result);
    }


template <typename T>
T Matern_kernel(const T* x, const unsigned int dimensions, const T sigma, const unsigned int n,
const T* Matern_comb_number)
    {
    unsigned int i;
    unsigned int k;
    T part_1 = 0; // sum(x^2)^1/2
    T z;
    T Matern_part;
    
    for (i = 0; i < dimensions; i++)
        part_1 = part_1 + (x[i] * x[i]);
    
    part_1 = std::sqrt(part_1);
    z = 2 * part_1/sigma;
    // Horner scheme for sum_{k=0}^{n} c_k z^(n-k)
    Matern_part = Matern_comb_number[0];
    for (k = 1; k <= n; k++)
        Matern_part = Matern_part * z + Matern_comb_number[k];
    
    return(std::exp(-part_1/sigma) * Matern_part);
    }


template <typename T>
unsigned int Matern_coefficient(const unsigned int n, T* comb_number) // values of (n + k)!/(2 n)! * n!/(k!(n-k)!), k = 0..n
    {
    unsigned int k;
    
    // downward recurrence c_n = 1, c_{k-1} = c_k k/((n+k)(n-k+1)), no factorials (no overflow)
    comb_number[n] = 1;
    for (k = n; k > 0; k--)
        comb_number[k - 1] = comb_number[k] * T(k)/(T(n + k) * T(n - k + 1));
    return(0);
    }



template <typename T>
REG<T>::REG(const T* x, const T* y, const T* params, const unsigned int data_size, const unsigned int x_dimensions, const string kernel)
    {
    Init(x, y , params, data_size, x_dimensions, kernel);
    }


template <typename T>
REG<T>::REG(const string x_file, const string y_file, const string param_file)
    {
    unsigned int i;
    ifstream xfile(x_file);
    ifstream yfile(y_file);
    ifstream paramfile(param_file);
    string line;
    T line_value;
    // parameters of model
    unsigned int data_size;
    unsigned int x_dimensions;
    string kernel;
    
    REG_param_file = new T[4];
    if(!REG_param_file) laerror("allocation failure in REG constructor");
    // loading parameters for regression model from param_file
    if (paramfile.is_open())
        {
        if (getline(paramfile, line))
            data_size = stoi(line);
        else
            return;
        
        if (getline(paramfile, line))
            x_dimensions = stoi(line);
        else
            return;
        
        if (getline(paramfile, line))
            kernel = line;
        else
            return;
        
        if (getline(paramfile, line))
            REG_param_file[0] = stod(line);
        else
            return;
        
        if (getline(paramfile, line))
            REG_param_file[1] = stod(line);
        else
            return;
        
        if (getline(paramfile, line))
            REG_param_file[2] = stod(line);
        else
            return;
            
        if (kernel == "Matern")
            {
            if (getline(paramfile, line))
                REG_param_file[3] = static_cast<unsigned int>(stod(line) + 0.5);
            else
                return;
            }
        }
        REG_x_file = new T[data_size * x_dimensions];
        REG_y_file = new T[data_size];
	if(!REG_x_file || !REG_y_file) laerror("allocation failure 2 in REG constructor");

    // loading x and y data for regression model from files
    if (xfile.is_open() and yfile.is_open())
        {
        for (i = 0; i < data_size * x_dimensions; i++)
            {
            if (getline(xfile, line))
                {
                line_value = stod(line);
                REG_x_file[i] = line_value;
                }
            else
                return;
            }
        for (i = 0; i < data_size; i++)
            {
            if (getline(yfile, line))
                {
                line_value = stod(line);
                REG_y_file[i] = line_value;
                }
            else
                return;
            }
        xfile.close();
        yfile.close();
        paramfile.close();
        }
    else
        return;
    
    Init(REG_x_file, REG_y_file, REG_param_file, data_size, x_dimensions, kernel);
    return;
    }


template <typename T>
void REG<T>::Init(const T* x, const T* y, const T* params, const unsigned int data_size, const unsigned int x_dimensions,
const string kernel)
    {
    unsigned int i;
    // Setting type of kernel
    REG_kernel_type = kernel;
    if (kernel != "Gaussian" and kernel != "Laplacian" and kernel != "Exponential" and kernel != "Matern")
        return;
    if (kernel == "Gaussian")
        REG_kernel_id = 0;
    if (kernel == "Laplacian")
        REG_kernel_id = 1;
    if (kernel == "Exponential")
        REG_kernel_id = 2;
    if (kernel == "Matern")
        REG_kernel_id = 3;
    // copy parameters and data
    REG_lambda= params[0];
    REG_alpha = params[1];
    if (REG_alpha > 1)
        REG_alpha = 1;
    if (REG_alpha < 0)
        REG_alpha = 0;
    REG_sigma = params[2];
    if (kernel == "Matern")
        REG_n = static_cast<unsigned int>(params[3] + 0.5);
    
    REG_count_kernels = data_size;
    REG_count_dimensions = x_dimensions;
    REG_aloc_init = false;
        {
        REG_params = new T[4];
	if(!REG_params) laerror("allocation failure in REG::Init");
        if (REG_kernel_id == 3)
            {
            for (i = 0; i < 4; i++)
                REG_params[i] = params[i];
            REG_Matern_comb_number_vec = NRVec<T>(REG_n + 1);
            REG_Matern_comb_number = REG_Matern_comb_number_vec.operator T*();
            }
        else
            {
            for (i = 0; i < 3; i++)
                REG_params[i] = params[i];
                
            REG_Matern_comb_number_vec = NRVec<T>(1);
            REG_Matern_comb_number = REG_Matern_comb_number_vec.operator T*();
            }
        }

    // raw pointers to the data of the vectors below are cached for speed; the vectors must therefore never be
    // reallocated: a |= b frees the old buffer of a (pointers would dangle), use a.storesubvector(0,b) (copy in place)
    // copy data
    REG_coordinates_vec = NRVec<T>(x, data_size * REG_count_dimensions);
    REG_coordinates = REG_coordinates_vec.operator T*();
    REG_y_vec = NRVec<T>(y , data_size);
    REG_Y = REG_y_vec.operator T*();
    // for fitting model
    REG_distances_mat = NRMat<T>(data_size, data_size*REG_count_dimensions); //difference vectors of all pairs
    REG_distances = REG_distances_mat.operator T*();
    REG_kernels_mat = NRMat<T>(data_size, data_size);
    REG_kernels = REG_kernels_mat.operator T*();
    REG_kernels_sums_vec = NRVec<T>(data_size);
    REG_kernels_sums = REG_kernels_sums_vec.operator T*();
    REG_weights_vec = NRVec<T>(data_size);
    REG_weights = REG_weights_vec.operator T*();
    // for predictions
    REG_coordinates_diff_vec = NRVec<T>(data_size*REG_count_dimensions);
    REG_coordinates_diff = REG_coordinates_diff_vec.operator T*();
    REG_kernels_diff_vec = NRVec<T>(data_size);
    REG_kernels_diff = REG_kernels_diff_vec.operator T*();
    REG_y_preds_vec = NRVec<T>(data_size);
    REG_y_preds = REG_y_preds_vec.operator T*();
    REG_y_delta_vec = NRVec<T>(data_size);
    REG_y_delta = REG_y_delta_vec.operator T*();
    REG_y_temp_vec = NRVec<T>(data_size);
    REG_y_temp = REG_y_temp_vec.operator T*();
    
    REG_aloc_init = true;
    
    // calculating Matern coefficients, if Matern kernel is selected
    if (REG_kernel_id == 3)
        Matern_coefficient(REG_n, REG_Matern_comb_number);
    
    REG_base_init = true;
    return;
    }



template <typename T>
T REG<T>::Kernel(const T* x, const unsigned int dimensions, const unsigned int kernel_id)
    {
    T result;
    
    switch (kernel_id) {
    case 0:
        result = Gaussian_kernel(x, dimensions, REG_sigma);
        break;
    case 1:
        result = Laplacian_kernel(x, dimensions, REG_sigma);
        break;
    case 2:
        result = Exponential_kernel(x, dimensions, REG_sigma);
        break;
    case 3:
        result = Matern_kernel(x, dimensions, REG_sigma, REG_n, REG_Matern_comb_number);
        break;
    }
    return(result);
    }


template <typename T>
unsigned int REG<T>::Build_REG_distances_mat()
    {
    unsigned int i, j, k;
    unsigned int mat_dimension, vec_dimension;
    // check the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    mat_dimension = REG_count_kernels;
    vec_dimension = REG_count_dimensions;
    // building the matrix of vectors coordinates differences between kernels
    for (i = 0; i < mat_dimension; i++)
        {
        for (j = i + 1; j < mat_dimension; j++)
            for (k = 0; k < vec_dimension; k++)
                {
                REG_distances[(i * (mat_dimension) + j) * vec_dimension + k] = REG_coordinates[i * vec_dimension + k] -
                REG_coordinates[j * vec_dimension + k];
                }
        }
    // copy calculated values under diagonal
    for (i = 0; i < mat_dimension; i++)
        {
        for (j = i + 1; j < mat_dimension; j++)
            for (k = 0; k < vec_dimension; k++)
                {
                REG_distances[(j * (mat_dimension) + i) * vec_dimension + k] =
                REG_distances[(i * (mat_dimension) + j) * vec_dimension + k];
                }
        }
    // copy 0 to diagonal
    for (i = 0; i < mat_dimension; i++)
        for (k = 0; k < vec_dimension; k++)
            REG_distances[(i * (mat_dimension + 1) * vec_dimension) + k] = 0;
    
    return(0);
    }


template <typename T>
unsigned int REG<T>::Build_REG_kernels_mat()
    {
    unsigned int i, j, k;
    unsigned int mat_dimension, vec_dimension;
    // check the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    // copy class parameters
    mat_dimension = REG_count_kernels;
    vec_dimension = REG_count_dimensions;
    // crete kernel coefficients matrix for gaussian kernels
    for (i = 0; i < mat_dimension; i++)
        for (j = i + 1; j < mat_dimension; j++)
            {
            REG_kernels[i * mat_dimension + j] = Kernel(&REG_distances[(i * (mat_dimension) + j) * vec_dimension],
            vec_dimension, REG_kernel_id);
            }
    // copy calculated values under diagonal
    for (i = 0; i < mat_dimension; i++)
        for (j = i + 1; j < mat_dimension; j++)
            REG_kernels[(j * (mat_dimension) + i)] = REG_kernels[(i * (mat_dimension) + j)];
    
    // copy 1 to diagonal
    for (i = 0; i < mat_dimension; i++)
        REG_kernels[i * (mat_dimension + 1)] = 1;
    
    return(0);
    }


template <typename T>
unsigned int REG<T>::Build_REG_kernels_mat_sums()
    {
    unsigned int i, j;
    unsigned int count_kernels;
    T sum;
    
    count_kernels = REG_count_kernels;
    for (i = 0; i < count_kernels; i++)
        {
        sum = 0;
        for (j = 0; j < count_kernels; j++)
            sum += REG_kernels[(i * count_kernels) + j];
        
        REG_kernels_sums[i] = sum;
        }
    return(0);
    }


template <typename T>
T REG<T>::REG_RMSE()
    {
    T delta_square;
    // check the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    T y_max = REG_y_vec.amax();
    T y_min = REG_y_vec.amin();
   
    REG_y_delta_vec.storesubvector(0, REG_y_preds_vec); //in place, not |= (cached raw pointer)
    REG_y_delta_vec -= REG_y_vec;
    REG_y_delta_vec *= REG_y_delta_vec;
    delta_square = REG_y_delta_vec.sum();
    delta_square = delta_square/REG_count_kernels;
    delta_square = std::sqrt(delta_square);
    return(delta_square);
    }


template <typename T>
T REG<T>::Loss_Function()
    {
    unsigned int count_x;
    T RMSE;
    // check the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    // calculating the root mean square error - RMSE
    RMSE = REG_RMSE();
    return (RMSE);
    }


template <typename T>
unsigned int REG<T>::REG_Fit(const T max_Loss, const unsigned int max_learning_cycles)
    { // the kernel ridge learning algorithm
    unsigned int i, j;
    T Loss;
    T Lasso_part;
    T kernel_ridge_part;
    unsigned int count_kernels;
    
    // check the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    // checking the input parameters
    if (max_learning_cycles == 0)
        {
        laerror("zero learning cycles");
        return(1);
        }
    if (max_Loss < 0)
        {
        laerror("negative value of maximum Loss function");
        return(1);
        }
    Lasso_part = REG_lambda * REG_alpha;
    kernel_ridge_part = REG_lambda * (1 - REG_alpha) * 0.5;
    count_kernels = REG_count_kernels;
    
    // initialization of REG_distances_mat and REG_kernels_mat if necessary
    if (REG_mat_init == false)
        {
        Build_REG_distances_mat();
        Build_REG_kernels_mat();
        Build_REG_kernels_mat_sums();
        REG_mat_init = true;
        }
    // initialitation of weight coefficients vector
    REG_weights_vec.storesubvector(0, REG_y_vec); //in place, not |= (cached raw pointer)
    // Linear learning cycle of weight coefficient for learning
    for (i = 0; i < max_learning_cycles; i++)
        {
        // calculating preds_vec vector of predicted y values
        REG_y_preds_vec.gemv(0, REG_kernels_mat, 'n', 1, REG_weights_vec);
        if (kernel_ridge_part != 0)
            for (j = 0; j < REG_count_kernels; j++)
                REG_y_preds[j] *= (REG_kernels_sums[j] + kernel_ridge_part)/REG_kernels_sums[j];
            
        if (Lasso_part != 0)
            REG_y_preds_vec += Lasso_part;
        
        // calculating new weights
        REG_y_temp_vec.storesubvector(0, REG_y_vec); //in place, not |= (cached raw pointer)
        REG_y_temp_vec /= REG_y_preds_vec;
        REG_weights_vec *= REG_y_temp_vec;
        // checking the Loss_Function value
        if (max_Loss > 0)
            {
            Loss = Loss_Function();
            if (Loss < max_Loss)
               break;
            }
        }
    return(0);
    }


template <typename T>
unsigned int REG<T>::REG_Fit_grad(const T max_Loss, const unsigned int max_learning_cycles, const T learning_rate)
    {
    unsigned int i, j, k;
    unsigned int count_x;
    T Loss;
    T delta_RMSE;
    // for optimizing calculation of Kernel ridge part of Loss function difference
    T kernel_ridge_part;
    T delta_KRR_part;
    // for optimizing calculation of Lasso part of Loss function difference
    T Lasso_part;
    T delta_Lasso_part;
    T sqrt_count_x;
    
    count_x = REG_count_kernels;
    // check the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    // Checking the input parameters
    if (max_learning_cycles == 0)
        {
        laerror("zero learning cycles");
        return(1);
        }
    if (max_Loss < 0)
        {
        laerror("negative value of maximum Loss function");
        return(1);
        }
    // initialization of REG_distances_mat and REG_kernels_mat if necessary
    if (REG_mat_init == false)
        {
        Build_REG_distances_mat();
        Build_REG_kernels_mat();
        Build_REG_kernels_mat_sums();
        REG_mat_init = true;
        }
    // initialitation of weight coefficients vector
    REG_weights_vec.storesubvector(0, REG_y_vec); //in place, not |= (cached raw pointer)
    kernel_ridge_part = REG_lambda * (1.00 - REG_alpha) * 0.5;
    Lasso_part = REG_lambda * REG_alpha;
    sqrt_count_x = std::sqrt(count_x);
    // gradient descent Learning cycle of weight coefficient for learning
    for (i = 0; i < max_learning_cycles; i++)
        {
        // calculating preds_vec vector of predicted y values
        REG_y_preds_vec.gemv(0, REG_kernels_mat, 'n', 1, REG_weights_vec);
        if (kernel_ridge_part != 0)
            for (j = 0; j < count_x; j++)
                REG_y_preds[j] *= (REG_kernels_sums[j] + kernel_ridge_part)/REG_kernels_sums[j];
            
        if (Lasso_part != 0)
            REG_y_preds_vec += Lasso_part;
        // calculating difference of root mean square error - RMSE
        for (j = 0; j < count_x; j++)
            {
            delta_RMSE = 0;
            for (k = 0; k < count_x; k++)
                delta_RMSE += (REG_y_preds[k] - REG_Y[k]) * REG_kernels[j * count_x + k];
            
            delta_RMSE /=sqrt_count_x;
            // calculating new weights
            REG_weights[j] /= (1 - learning_rate * delta_RMSE);
            }
        // convergence criteria check
        if (max_Loss > 0)
            {
            Loss = Loss_Function();
            if (Loss < max_Loss)
               break;
            }
        }
    return(0);
    }


template <typename T>
unsigned int REG<T>::REG_Save_fitted(const string model_file)
    {
    unsigned int i;
    ofstream modelfile(model_file);
    string line;
    
    // checking the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    // save model to file
    if (modelfile.is_open())
        for (i = 0; i < REG_count_kernels; i++)
            {
            line = to_string(REG_weights[i]);
            modelfile << line << '\n';
            }
    modelfile.close();
    
    return(0);
    }


template <typename T>
unsigned int REG<T>::REG_Load_fitted(const string model_file)
    {
    unsigned int i;
    ifstream modelfile(model_file);
    string line;
    T line_value;
    // check the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    // initialization of REG_distances_mat and REG_kernels_mat if necessary
    if (REG_mat_init == false)
        {
        Build_REG_distances_mat();
        Build_REG_kernels_mat();
        REG_mat_init = true;
        }
    // load stored weights
    if (modelfile.is_open())
        {
        for (i = 0; i < REG_count_kernels; i++)
            if (getline(modelfile, line))
                {
                line_value = stod(line);
                REG_weights[i] = line_value;
                }
        REG_y_preds_vec.gemv(0, REG_kernels_mat, 'n', 1, REG_weights_vec);
        }
    return(0);
    }


template <typename T>
unsigned int REG<T>::REG_Get_predictions(const T* x, T* y_preds, const unsigned int data_size)
    {
    unsigned int i, j, k;
    T pred_value;
    T Lasso_part;
    T kernel_ridge_part;
    T sum_kernels_diff;
    T weights_sum;
    // check the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    Lasso_part = REG_lambda * REG_alpha;
    kernel_ridge_part = REG_lambda * (1 - REG_alpha) * 0.5;
    // loop for predicting new values for each x 
    for (i = 0; i < data_size; i++) // For every x value
        {
        pred_value = 0;
        // calculating the REG_coordinates_vec
        for (j = 0; j < REG_count_kernels; j++) // For every kernel
            {
            for (k = 0; k < REG_count_dimensions; k++) // For every x coordinate
                {
                REG_coordinates_diff[(j * REG_count_dimensions) + k] = x[(i * REG_count_dimensions) + k] -
                REG_coordinates[(j * REG_count_dimensions) + k];
                }
            }
        // calculating the REG_kernels_vec
        if (kernel_ridge_part != 0)
            {
            sum_kernels_diff = 0;
            for (j = 0; j < REG_count_kernels; j++) // For every kernel
                {
                REG_kernels_diff[j] = Kernel(&REG_coordinates_diff[(j * REG_count_dimensions)], REG_count_dimensions, REG_kernel_id);
                pred_value += REG_kernels_diff[j] * REG_weights[j];
                sum_kernels_diff += REG_kernels_diff[j];
                }
            if (sum_kernels_diff != 0)
                pred_value *= (sum_kernels_diff + kernel_ridge_part)/sum_kernels_diff;
            }
        else
            for (j = 0; j < REG_count_kernels; j++) // For every kernel
                {
                REG_kernels_diff[j] = Kernel(&REG_coordinates_diff[(j * REG_count_dimensions)], REG_count_dimensions, REG_kernel_id);
                pred_value += REG_kernels_diff[j] * REG_weights[j];
                }
        pred_value += Lasso_part;
        // storing y prediction into memory
        y_preds[i] = pred_value;
        }
    return(0);
    }


template <typename T>
unsigned int REG<T>::REG_Get_Score(const T* x, T* y_preds,  const unsigned int data_size, const T* y, T* score)
    {
    unsigned int i;
    T prediction;
    // check the initialization
    if (REG_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    REG_Get_predictions(x, y_preds, data_size);
    prediction = 0;
    for (i = 0; i < data_size; i++)
        prediction += (y[i] - y_preds[i]) * (y[i] - y_preds[i]);
    
    score[0] = prediction;
    return(0);
    }


template <typename T>
unsigned int REG<T>::REG_Set_params(const T* params, const string kernel)
    {
    const unsigned int old_kernel_id = REG_kernel_id;
    const unsigned int old_n = (REG_kernel_id == 3) ? REG_n : 0; // REG_n is set only for Matern
    const T old_sigma = REG_sigma;
    // setting type of kernel
    REG_kernel_type = kernel;
    if (kernel == "Gaussian")
        REG_kernel_id = 0;
    if (kernel == "Laplacian")
        REG_kernel_id = 1;
    if (kernel == "Exponential")
        REG_kernel_id = 2;
    if (kernel == "Matern")
        REG_kernel_id = 3;
    // copy parameters and data
    REG_lambda = params[0];
    REG_alpha = params[1];
    if (REG_alpha > 1)
        REG_alpha = 1;
    if (REG_alpha < 0)
        REG_alpha = 0;
    REG_sigma = params[2];
    if (kernel == "Matern")
        REG_n = static_cast<unsigned int>(params[3] + 0.5);
    
    // calculating Matern coefficients, if Matern kernel is selected
    if (REG_kernel_id == 3)
        {
        REG_Matern_comb_number_vec.resize(REG_n + 1);
        REG_Matern_comb_number = REG_Matern_comb_number_vec.operator T*();
        Matern_coefficient(REG_n, REG_Matern_comb_number);
        }
    // kernel matrices depend on kernel, sigma and n (not on lambda, alpha): rebuild them in the next fit
    if (REG_kernel_id != old_kernel_id || REG_sigma != old_sigma || (REG_kernel_id == 3 && REG_n != old_n))
        REG_mat_init = false;
    return(0);
    }


template <typename T>
unsigned int REG<T>::REG_Get_params(T* params, string* kernel)
    {
    kernel[0] = REG_kernel_type;
    params[0] = REG_lambda;
    params[1] = REG_alpha;
    params[2] = REG_sigma;
    
    if (REG_kernel_id == 3)
        params[3] = REG_n;
    
    return(0);
    }


template <typename T>
unsigned int REG<T>::REG_Get_weight_coeff(T* weight)
    {
    unsigned int i;
    
    for (i = 0; i < REG_count_kernels; i++)
        weight[i] = REG_weights[i];
    
    return(0);
    }


template <typename T>
REG<T>::~REG()
    {
    // dealocating memory to avoid memory leaks
    if (REG_params != nullptr)
        delete[] REG_params;
    if (REG_x_file != nullptr)
        delete[] REG_x_file;
    if (REG_y_file != nullptr)
        delete[] REG_y_file;
    if (REG_param_file != nullptr)
        delete[] REG_param_file;
    if (REG_fit_file != nullptr)
        delete[] REG_fit_file;
    }

//enforce instantiation

#define INSTANTIZE(T) \
template T Gaussian_kernel(const T* x, const unsigned int dimensions, const T sigma); \
template T Laplacian_kernel(const T* x, const unsigned int dimensions, const T sigma); \
template T Exponential_kernel(const T* x, const unsigned int dimensions, const T sigma); \
template T Matern_kernel(const T* x, const unsigned int dimensions, const T sigma, const unsigned int n, const T* Matern_comb_number); \
template unsigned int Matern_coefficient(const unsigned int n, T* comb_number); \


INSTANTIZE(double)

} // end of namespace
template class REG<double>;
