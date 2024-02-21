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

#include "regsurf.h"

namespace LA {

template <typename T>
unsigned int Build_s_RE_descriptor_mat(const unsigned int count_particles,
const unsigned int count_geometries, const T* distances, T* desc, const T* reference_geometry)
    {
    unsigned int i, j, k;
    T distance;
    T distance_ref;
    // the RE descriptor
    for (i = 0; i < count_geometries; i++)
        {
        // calculate upper part of descriptor matrix
        for (j = 0; j < count_particles; j++)
            for (k = j + 1; k < count_particles; k++)
                {
                distance = distances[(i * count_particles * count_particles) + (j * count_particles) + k];
                distance_ref = reference_geometry[(j * count_particles) + k];
                desc[(i * count_particles * count_particles) + (j * count_particles) + k] = distance_ref/distance;
                }
        // copy 0 to diagonal
        for (j = 0; j < count_particles; j++)
            desc[(i * count_particles * count_particles) + j * (count_particles + 1)] = 0;
        // copy upper part of descriptor matrix under diagonal
        for (j = 0; j < count_particles; j++)
            for (k = 0; k < j; k++)
                desc[(i * count_particles * count_particles) + (j * count_particles) + k] =
                desc[(i * count_particles * count_particles) + (k * count_particles) + j];
        }
    return(0);
    }


template <typename T>
unsigned int Build_s_coulomb_matrix_descriptor_mat(const unsigned int count_particles,
const unsigned int count_geometries, const T* distances, T* desc, const unsigned int* Z)
    {
    unsigned int i, j, k;
    T distance;
    // the coulomb matrix descriptor
    for (i = 0; i < count_geometries; i++)
        {
        // calculate upper part of descriptor matrix
        for (j = 0; j < count_particles; j++)
            for (k = j + 1; k < count_particles; k++)
                {
                distance = distances[(i * count_particles * count_particles) + (j * count_particles) + k];
                desc[(i * count_particles * count_particles) + (j * count_particles) + k] = (Z[j] * Z[k])/distance;
                }
        // calculate the diagonal
        for (j = 0; j < count_particles; j++)
            desc[(i * count_particles * count_particles) + j * (count_particles + 1)] = std::pow(Z[j], 2.4);
        // copy upper part of descriptor matrix under diagonal
        for (j = 0; j < count_particles; j++)
            for (k = 0; k < j; k++)
                desc[(i * count_particles * count_particles) + (j * count_particles) + k] =
                desc[(i * count_particles * count_particles) + (k * count_particles) + j];
        }
    return(0);
    }


template <typename T>
unsigned int Build_s_inverse_pairs_descriptor_mat(const unsigned int count_particles,
const unsigned int count_geometries, const T* distances, T* desc)
    {
    unsigned int i, j, k;
    T distance;
    // the inverse pairs descriptor
    for (i = 0; i < count_geometries; i++)
        {
        // calculate upper part of descriptor matrix
        for (j = 0; j < count_particles; j++)
            for (k = 0; k < count_particles; k++)
                {
                distance = distances[(i * count_particles * count_particles) + (j * count_particles) + k];
                desc[(i * count_particles * count_particles) + (j * count_particles) + k] = 1.00/distance;
                }
        // copy 0 to diagonal
        for (j = 0; j < count_particles; j++)
            desc[(i * count_particles * count_particles) + j * (count_particles + 1)] = 0;
        // copy upper part of descriptor matrix under diagonal
        for (j = 0; j < count_particles; j++)
            for (k = 0; k < j; k++)
                desc[(i * count_particles * count_particles) + (j * count_particles) + k] =
                desc[(i * count_particles * count_particles) + (k * count_particles) + j];
        }
    return(0);
    }


// The kernel regression of surfaces of geometries in in 1D, 2D, 3D and multi-D space
// and computing of ML-NEA absorption spectra

template <typename T>
REGSURF<T>::REGSURF(const unsigned int count_dimensions, const unsigned int count_kernels, const unsigned int count_geometries,
const unsigned int count_levels, const string descriptor, const string kernel, const T* params,  const T* geometries,
const T* values, const unsigned int* Z)
    {
    Init(count_dimensions, count_kernels, count_geometries, count_levels, descriptor, kernel, params, geometries, values, Z);
    }


template <typename T>
REGSURF<T>::REGSURF(const string geometries_file, const string y_file, const string param_file, const string Z_file)
    {
    unsigned int i, j, k, l;
    unsigned int line_size, numbers_size;
    ifstream geomfile(geometries_file);
    ifstream yfile(y_file);
    ifstream paramfile(param_file);
    ifstream Zfile(Z_file);
    string line;
    T line_value;
    // parameters of model
    unsigned int count_dimensions;
    unsigned int count_kernels;
    unsigned int count_geometries;
    unsigned int count_levels;
    string kernel;
    string descriptor;
    const string numbers = "+-0123456789.,";
    string readed_number;
    bool read_switch;
    bool decimal_switch;
    
    
    // loading parameters for regression model from parameters_file
    if (paramfile.is_open())
        {
        // load fixed paraeters of model
        if (getline(paramfile, line))
            count_dimensions = stoi(line);
        else
            {
            laerror("file with model paramaters not valid");
            return;
            }
        if (getline(paramfile, line))
            count_kernels = stoi(line);
        else
            
            {
            laerror("file with model paramaters not valid");
            return;
            }
        if (getline(paramfile, line))
            count_geometries = stoi(line);
        else
            {
            laerror("file with model paramaters not valid");
            return;
            }
        if (getline(paramfile, line))
            count_levels = stoi(line);
        else
            {
            laerror("file with model paramaters not valid");
            return;
            }
        // load flexible parameters of model
        if (getline(paramfile, line))
            descriptor = line;
        else
            {
            laerror("file with model paramaters not valid");
            return;
            }
        if (getline(paramfile, line))
            kernel = line;
        else
            {
            laerror("file with model paramaters not valid");
            return;
            }
        // alloacate memory for parameters file 
            {
            s_param_file = new T[count_levels * 2 + 2];
            if(!s_param_file) laerror("not allocated memory for parameters");
            }
        
        for (i = 0; i < ((2 * count_levels) + 1); i++)
            {
            if (getline(paramfile, line))
                s_param_file[i] = stod(line);
            else
                {
                laerror("file with model paramaters not valid");
                return;
                }
            }
        
        if (kernel == "Matern")
            {
            if (getline(paramfile, line))
                s_param_file[(2 * count_levels) + 1] = static_cast<unsigned int>(stod(line) + 0.5);
            else
                {
                laerror("file with model paramaters not valid");
                return;
                }
            }
        paramfile.close();
        }
    // alocating memory for y data for regression model from files
        {
        s_geom_file = new T[count_dimensions * count_kernels * count_geometries];
	if(!s_geom_file) laerror("allocation error of s_geom_file");
        s_y_file = new T[count_geometries * count_levels];
	if(!s_y_file) laerror("allocation error of s_y_file");
        if (descriptor == "Coulomb_matrix" or Z_file != "")
	    {
            s_Z_file = new unsigned int[count_kernels];
	    if(!s_Z_file)  laerror("allocation error of s_Z_file");
	    }
	else s_Z_file=NULL;
        s_y_min = new unsigned int[count_levels];
	if(!s_y_min) laerror("allocation error of s_y_min");
        }
    // loading geometrical data for regression model from file
    l = 0;
    numbers_size = numbers.size();
    if (geomfile.is_open())
        {
        for (i = 0; i < 2 * count_dimensions * count_kernels * count_geometries; i++)
            {
            if (getline(geomfile, line))
                {
                read_switch = false;
                decimal_switch = false;
                readed_number = "";
                line_size = line.size();
                for (j = 0; j < line_size; j++)
                    {
                    read_switch = false;
                    for (k = 0; k < numbers_size; k++)
                        {
                        if (line[j] == numbers[k])
                            {
                            readed_number.append(line, j, 1);
                            read_switch = true;
                            if (line[j] == numbers[12] or line[j] == numbers[13])
                                decimal_switch = true;
                            }
                        }
                    if (read_switch == false)
                        {
                        if (decimal_switch == true and readed_number != "")
                            {
                            line_value = stod(readed_number);
                            s_geom_file[l] = line_value;
                            l++;
                            }
                        readed_number = "";
                        read_switch = false;
                        decimal_switch = false;
                        }
                    }
                if (decimal_switch == true and readed_number != "")
                    {
                    line_value = stod(readed_number);
                    s_geom_file[l] = line_value;
                    l++;
                    }
                }
            if (l == count_dimensions * count_kernels * count_geometries)
                break;
            }
        geomfile.close();
        }
    else
        {
        laerror("file with geometries not open");
        return;
        }
        
    if (l < count_dimensions * count_kernels * count_geometries)
        {
        laerror("incomplete file with geometries");
        return;
        }
    // loading y data for model from file
    if (yfile.is_open())
        {
        for (i = 0; i < count_geometries * count_levels; i++)
            {
            if (getline(yfile, line))
                {
                line_value = stod(line);
                s_y_file[i] = line_value;
                }
            else
                {
                yfile.close();
                laerror("incomplete file with y values");
                return;
                }
            }
        }
    else
        {
        laerror("file with y values not open");
        return;
        }
    // loading Z proton numbers of atoms
    if (s_Z_file != nullptr and Z_file != "")
        {
        if (Zfile.is_open())
            {
            for (i = 0; i < count_kernels; i++)
                {
                if (getline(Zfile, line))
                    s_Z_file[i] = stoi(line);
                else
                    {
                    laerror("incomplete file with Z quantum numbers ");
                    return;
                    }
                }
            Zfile.close();
            }
        else
            {
            laerror("file with Z quantum numbers not open");
            return;
            }
        }
    // callling init function with loaded parameters, y and geometries data
    Init(count_dimensions, count_kernels, count_geometries, count_levels, descriptor, kernel, s_param_file, s_geom_file, s_y_file,
    s_Z_file);
    }


template <typename T>
void REGSURF<T>::Init(const unsigned int count_dimensions, const unsigned int count_kernels, const unsigned int count_geometries,
    const unsigned int count_levels, const string descriptor, const string kernel, const T* params,  const T* geometries,
    const T* values, const unsigned int* Z)
    {
    unsigned int i, j;
    T min_value;
    
    s_kernel_type = kernel;
    if (kernel != "Gaussian" and kernel != "Laplacian" and kernel != "Exponential" and kernel != "Matern")
        {
        laerror("invalid type of kernel");
        return;
        }
    if (kernel == "Gaussian")
        s_kernel_id = 0;
    if (kernel == "Laplacian")
        s_kernel_id = 1;
    if (kernel == "Exponential")
        s_kernel_id = 2;
    if (kernel == "Matern")
        s_kernel_id = 3;
        
    s_descriptor_type = descriptor;
    if (descriptor == "RE")
        s_descriptor_id = 0;
    if (descriptor == "Coulomb_matrix")
        s_descriptor_id = 1;
    if (descriptor == "Inverse_pairs")
        s_descriptor_id = 2;
    
    s_count_kernels = count_kernels;
    s_count_dimensions = count_dimensions;
    s_count_geometries = count_geometries;
    s_count_levels = count_levels;
    
    // copy parameters and data
        {
        s_lambda = new T[count_levels];
        s_alpha = new T[count_levels];
	if(!s_lambda || !s_alpha) laerror("allocation error in REGSURF::Init");
        }
    for (i = 0; i < count_levels; i++)
        s_lambda[i] = params[i];
   
    for (i = count_levels; i < (2 * count_levels); i++)
        {
        s_alpha[i - count_levels] = params[i];
        if (s_alpha[i] > 1)
            s_alpha[i] = 1;
        if (s_alpha[i] < 0)
            s_alpha[i] = 0;
        }
    s_sigma = params[(2 * count_levels)];
    if (kernel == "Matern")
        s_n = static_cast<unsigned int>(params[(2 * count_levels) + 1] + 0.5);
   
    s_aloc_init = false;
        { // for loaded parameters
        if (s_kernel_id == 3)
            {
            s_params = new T[(2 * count_levels) + 2];
	    if(!s_params) laerror("allocation error in REGSURF::Init");
            s_Matern_comb_number = new T[s_n];
	    if(!s_Matern_comb_number) laerror("allocation error in REGSURF::Init");
            for (i = 0; i < ((2 * count_levels) + 2); i++)
                s_params[i] = params[i];
            }
        else
            {
            s_params = new T[(2 * count_levels) + 1];
            if(!s_params) laerror("allocation error in REGSURF::Init");
            for (i = 0; i < ((2 * count_levels) + 1); i++)
                s_params[i] = params[i];
            }
        
        }
    // creating the list of numbers of geometries with minima of PES
    for (i = 0; i < count_levels; i++)
        {
        min_value = values[i * count_geometries];
        s_y_min[i] = 0;
        for (j = 1; j < count_geometries; j++)
            if (values[(i * count_kernels) + j] < min_value)
                {
                min_value = values[(i * count_kernels) + j];
                s_y_min[i] = j;
                }
        }
    // NRMat for input loaded geometries [s_count_dimensions * s_count_kernels, s_count_geometries]
    s_geometries_mat = NRMat<T>(geometries, count_dimensions * count_kernels, count_geometries);
    // NRMat for vector of y of geomeries for surface [s_count_geometries, s_count_levels]
    s_y_mat = NRMat<T>(values, count_levels, count_geometries);
    // NRVec for vector of Z for coulomb matrix descriptor [s_count_kernels]
    if (Z != nullptr)
        s_Z_vec = NRVec<unsigned int>(Z, count_kernels);
        
    // NRMat for coordinate distances for geometries [s_count_kernels, s_count_kernels * s_count_geometries]
    s_distances_mat = NRMat<T>(count_kernels, count_kernels * count_geometries);
    // NRMat for matrix of descriptors between kernels in geometry [s_count_kernels, s_count_kernels * s_count_geometries]
    s_desc_mat = NRMat<T>(count_kernels, count_kernels * count_geometries);
    // NRVec of NRMat for matrix of kernel coefficients for kernels in geometries
    // [s_count_kernels, s_count_kernels * s_count_geometries]
    s_kernels_vec_mat = NRVec<NRMat<T>>(count_geometries);
    for (i = 0; i < count_geometries; i++)
        s_kernels_vec_mat[i] = NRMat<T>(count_kernels, count_kernels);
    
    // NRVec for sums of kernels rows for base geometry [s_count_kernels]
    s_kernels_rows_sums_vec_vec = NRVec<NRVec<T>>(count_geometries);
    for (i = 0; i < count_geometries; i++)
        s_kernels_rows_sums_vec_vec[i] = NRVec<T>(count_kernels);
    // NRVec of NRVec for differences of sums kernels rows between base and other geometries [s_count_kernels, s_count_geometries]
    s_kernels_diff_vec_vec = NRVec<NRVec<T>>(count_geometries);
    for (i = 0; i < count_geometries; i++)
        s_kernels_diff_vec_vec[i] = NRVec<T>(count_kernels);
    // NRVec for differences of sums kernels rows and columns between base and other geometries [s_count_geometries]
    s_kernels_diff_sums_vec = NRVec<T>(count_geometries);
    
    // NRVec of NRVec for vector of weight coefficients of surface kernels in surface [s_count_kernels, s_count_levels]
    s_weights_vec_vec = NRVec<NRVec<T>>(count_levels);
    for (i = 0; i < count_levels; i++)
        s_weights_vec_vec[i] = NRVec<T>(count_kernels);
    
    // NRVec for vector of y of geomeries for surface [s_count_geometries]
    s_y_vec = NRVec<T>(s_y_mat.operator T*(), count_geometries);
    // NRVec for vector of y predictions of geomeries for surface [s_count_geometries]
    s_y_preds_vec = NRVec<T>(count_geometries);
    // NRVec for vector of delta y values for surface [s_count_geometries]
    s_y_delta_vec = NRVec<T>(count_geometries);
    // NRVec for vector of temporary y values for surface [s_count_geometries]
    s_y_temp_vec = NRVec<T>(count_kernels);
    // NRVec for vector of partial temporary y values for geometry [s_count_kernels]
    s_y_temp_part_vec = NRVec<T>(count_kernels);
    
    // NRMat for distances in geometry for predictions [s_count_kernels, s_count_kernels]
    s_distances_pred_mat = NRMat<T>(s_count_kernels, s_count_kernels);
    // NRMat for matrix of descriptors between kernels in geometry for predictions [s_count_kernels, s_count_kernels]
    s_desc_pred_mat = NRMat<T>(s_count_kernels, s_count_kernels);
    // NRVec of NRMat for matrix of kernel coefficients for kernels in geometry for predictions [count_kernels * count_kernels]
    s_kernels_pred_vec_mat = NRVec<NRMat<T>>(1);
    s_kernels_pred_vec_mat[0] = NRMat<T>(s_count_kernels, s_count_kernels);
    
    // for loaded geometries and surfaces
    s_geometries = s_geometries_mat.operator T*();
    s_y = s_y_mat.operator T*();
    // NRVec for vector of Z for coulomb matrix descriptor 
    s_Z = s_Z_vec.operator unsigned int*();
    // for computing weights coefficiensts of kernels and making predictions
    s_distances = s_distances_mat.operator T*();
    s_desc = s_desc_mat.operator T*();
    s_distances = s_distances_mat.operator T*();
    s_kernels = s_kernels_vec_mat[0].operator T*();
    s_kernels_rows_sums = s_kernels_rows_sums_vec_vec[0].operator T*();
    s_kernels_diff = s_kernels_diff_vec_vec[0].operator T*();
    s_kernels_diff_sums = s_kernels_diff_sums_vec.operator T*();
    s_weights = s_weights_vec_vec[0].operator T*();
        
    s_y_preds = s_y_preds_vec.operator T*();
    s_y_delta = s_y_delta_vec.operator T*();
    s_y_temp = s_y_temp_vec.operator T*();
    s_y_temp_part = s_y_temp_part_vec.operator T*();
    s_distances_pred = s_distances_pred_mat.operator T*();
    s_desc_pred = s_desc_pred_mat.operator T*();
    s_kernels_pred = s_kernels_pred_vec_mat[0].operator T*();
    s_aloc_init = true;
    // calculating Matern coefficients, if Matern kernel is selected
    if (s_kernel_id == 3)
        Matern_coefficient(s_n, s_Matern_comb_number);
    
    s_base_init = true;
    return;
    }


template <typename T>
T REGSURF<T>::Kernel(const T* x, const unsigned int dimensions, const unsigned int kernel_id)
    {
    T result;
    
    switch (kernel_id) {
    case 0:
        result = Gaussian_kernel(x, dimensions, s_sigma);
        break;
    case 1:
        result = Laplacian_kernel(x, dimensions, s_sigma);
        break;
    case 2:
        result = Laplacian_kernel(x, dimensions, s_sigma);
        break;
    case 3:
        result = Matern_kernel(x, dimensions, s_sigma, s_n, s_Matern_comb_number);
        break;
    }
    return(result);
    }



template <typename T>
unsigned int REGSURF<T>::Build_s_distances_mat(const unsigned int count_geometries, const T* geometries, T* distances)
    {
    unsigned int i, j, k, l;
    unsigned int count_dimensions, count_kernels;
    T distance;
    
    count_dimensions = s_count_dimensions;
    count_kernels = s_count_kernels;
    // filling the coordinate distances matrix
    if (s_base_init == true)
        for (i = 0; i < count_geometries; i++)
            for (j = 0; j < count_kernels; j++)
                for (k = 0; k < count_kernels; k++)
                    {
                    distance = 0;
                    for (l = 0; l < count_dimensions; l++)
                        {
                        distance += std::pow(geometries[(i * count_kernels * count_dimensions) + (j * count_dimensions) + l] - 
                                        geometries[(i * count_kernels * count_dimensions) + (k * count_dimensions) + l], 2);
                        }
                    distances[(i * count_kernels * count_kernels) + (j * count_kernels) + k] = std::pow(distance, 0.5);
                    }
    else
        {
        laerror("regression model not initialized");
        return(1);
        }
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::Build_s_descriptor_mat(const unsigned int count_geometries, const T* distances, T* desc)
    {
    unsigned int i, j, k, l;
    unsigned int count_kernels, count_dimensions;
    T distance, distance_ref;
    T descriptor;
    
    count_dimensions = s_count_dimensions;
    count_kernels = s_count_kernels;
    // filling the descriptor matrix
    if (s_base_init == true)
        {
        // the RE descriptor
        switch (s_descriptor_id) {
        case 0:
            Build_s_RE_descriptor_mat(s_count_kernels, count_geometries, distances, desc, s_distances);
            break;
        case 1:
            Build_s_coulomb_matrix_descriptor_mat(s_count_kernels, count_geometries, distances, desc, s_Z);
            break;
        case 2:
            Build_s_inverse_pairs_descriptor_mat(s_count_kernels, count_geometries, distances, desc);
            break;
        case 3:
            extern_desc(s_count_kernels, count_geometries, distances, desc);
            break;
        laerror("unknown decsriptor identification number");
        return(1);
        }
        }
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::Build_s_kernels_mat(const unsigned int count_geometries, const T* desc, NRVec<NRMat<T>> &kernels_vec_mat)
    {
    unsigned int i, j, k;
    unsigned int  count_kernels;
    T descriptor_difference;
    T* matrix;
    
    count_kernels = s_count_kernels;
    // create matrices of kernel coefficients for each geometry
    for (i = 0; i < count_geometries; i++)
        {
        matrix = kernels_vec_mat[i].operator T*();
        // creating upper part of matrix of kernel coefficinnts and the diagonal
        for (j = 0; j < count_kernels; j++)
            for (k = j + 1; k < count_kernels; k++)
                {
                descriptor_difference = desc[(i * count_kernels * count_kernels) + (j * count_kernels) + k] -
                s_desc[(j * count_kernels) + k];
                matrix[(j * count_kernels) + k] = Kernel(&descriptor_difference, 1, s_kernel_id);
                }
        // creating the diagonal
        for (j = 0; j < count_kernels; j++)
            {
            if (desc[(i * count_kernels * count_kernels) + j * (count_kernels + 1)] != 0)
                {
                descriptor_difference = desc[(i * count_kernels * count_kernels) + j * (count_kernels + 1)] -
                s_desc[(i * count_kernels * count_kernels) + j * (count_kernels + 1)];
                matrix[j * (count_kernels + 1)] = Kernel(&descriptor_difference, 1, s_kernel_id);;
                }
            else
                matrix[j * (count_kernels + 1)] = 0;
            }
        // copying upper part of matrix down
        for (j = 0; j < count_kernels; j++)
            for (k = 0; k < j; k++)
                matrix[(j * count_kernels) + k] = matrix[(k * count_kernels) + j];
        }
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::Build_s_kernels_mat_sums()
    {
    unsigned int i, j, k;
    unsigned int count_kernels, count_geometries;
    T sum;
    T* matrix;
    T* vector;
    T* temp;
    T* kernels_mat;
    
    count_geometries = s_count_geometries;
    count_kernels = s_count_kernels;
    temp = s_y_temp_vec.operator T*();
    
    // create matrix of kernel sums differences for each kernel geometry
    // fixed bug on rsum() function
    for (i = 0; i < count_geometries; i++)
        {
        matrix = s_kernels_vec_mat[i].operator T*();
        vector = s_kernels_rows_sums_vec_vec[i].operator T*();
        for (j = 0; j < count_kernels; j++)
            {
            sum = 0;
            for (k = 0; k < count_kernels; k++)
                sum += matrix[(j * count_kernels) + k];
        
            vector[j] = sum;
            }
        }
    
    for (i = 0; i < count_geometries; i++)
        {
        kernels_mat = s_kernels_vec_mat[i].operator T*();
        for (j = 0; j < count_kernels; j++)
            {
            sum = 0;
            for (k = 0; k < count_kernels; k++)
                sum += kernels_mat[(j * count_kernels) + k];
            
            temp[j] = sum;
            }
        s_kernels_diff_vec_vec[i] |= s_kernels_rows_sums_vec_vec[0];
        s_kernels_diff_vec_vec[i] -= s_y_temp_vec;
        }
    // create vector of kernel sums differences for each geometry
    for (i = 0; i < count_geometries; i++)
        s_kernels_diff_sums_vec[i] = s_kernels_diff_vec_vec[i].sum();
    
    // calculation of kernels matrix sum for base geometry
    s_kernels_sum_sum = T(count_kernels * count_kernels);
    return(0);
    }


template <typename T>
T REGSURF<T>::s_RMSE(const unsigned int level)
    {
    unsigned int i;
    T delta_square;
    T Lasso_part;
    T kernel_ridge_part;
    T sum_kernels_base;
    T y_pred_base;
    T y_pred;
    
    // check the initialization
    if (s_base_init == false)
        {
        laerror("regression model not initialized");
        return(1);
        }
    if (level >= s_count_levels)
        {
        laerror("level number out of model range");
        return(1);
        }
    
    if (s_descriptor_id == 1)
        sum_kernels_base = s_count_kernels * s_count_kernels;
    else
        sum_kernels_base = s_count_kernels * (s_count_kernels - 1.00);
    
    Lasso_part = s_lambda[level] * s_alpha[level];
    kernel_ridge_part = s_lambda[level] * (1 - s_alpha[level]) * 0.5 * s_count_kernels;
    // predictions for all geometries
    for (i = 0; i < s_count_geometries; i++)
        {
        s_y_temp_part_vec |= s_kernels_rows_sums_vec_vec[i];
        s_y_temp_part_vec *= s_weights_vec_vec[level];
        y_pred = s_y_temp_part_vec.sum();
        if (kernel_ridge_part != 0)
            y_pred *= (sum_kernels_base - s_kernels_diff_sums_vec[i] + kernel_ridge_part)/
            (sum_kernels_base - s_kernels_diff_sums_vec[i]);
        
        y_pred += Lasso_part;
        s_y_preds[i] = y_pred;
        }
    
    s_y_delta_vec |= s_y_preds_vec;
    // subtracting real values
    s_y_delta_vec -= s_y_mat.row(level);
    // the second power
    s_y_delta_vec *= s_y_delta_vec;
    // the sum
    delta_square = s_y_delta_vec.sum();
    // calculating RMSE
    delta_square = delta_square/T(s_count_geometries);
    delta_square = std::sqrt(delta_square);
    return(delta_square);
    }


template <typename T>
T REGSURF<T>::Loss_Function(const unsigned int level)
    {
    unsigned int i;
    unsigned int count_x;
    T RMSE;
    T KRR_part;
    T Lasso_part;
    // check the initialization
    if (s_base_init == false)
        {
        laerror("regression model not initialized");
        return(1);
        }
    // calculating the root mean square error - RMSE
    RMSE = s_RMSE(level);
    return (RMSE);
    }


template <typename T>
unsigned int REGSURF<T>::s_Fit(const T max_Loss, const unsigned int max_learning_cycles, const T learning_rate,
const bool element_invariant, unsigned int* kernel_classes)
    {
    unsigned int i, j, k, l, m;
    unsigned int count_kernels, count_geometries, count_levels;
    unsigned int element_count;
    unsigned int* kernel_classes_index;
    T init_weight;
    T y_pred, y_pred_2, y_pred_3;
    T y_diff;
    T Loss;
    T Lasso_part;
    T kernel_ridge_part;
    T element_sum;
    T sum_kernels_base;
    T y_pred_base;
    
    count_kernels = s_count_kernels;
    count_geometries = s_count_geometries;
    count_levels = s_count_levels;
    if (kernel_classes == nullptr)
        kernel_classes_index = s_Z;
    else
        kernel_classes_index = kernel_classes;
    // check the initialization
    if (s_base_init == false)
        {
        laerror("regression model not initialized");
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
    // initialization of s_distances_mat and s_kernels_mat if necessary
    if (s_distances_mat_init == false)
        {
        Build_s_distances_mat(s_count_geometries, s_geometries, s_distances);
        s_distances_mat_init = true;
        }
    if (s_descriptors_mat_init == false)
        {
        Build_s_descriptor_mat(s_count_geometries, s_distances, s_desc);
        s_descriptors_mat_init = true;
        }
    if (s_kernels_mat_init == false)
        {
        Build_s_kernels_mat(s_count_geometries, s_desc, s_kernels_vec_mat);
        Build_s_kernels_mat_sums();
        s_kernels_mat_init = true;
        }
    sum_kernels_base = s_kernels_rows_sums_vec_vec[0].sum();
    // the training loop for all surfaces
    for (i = 0; i < count_levels; i++)
        {
        Lasso_part = s_lambda[i] * s_alpha[i];
        kernel_ridge_part = s_lambda[i] * (1 - s_alpha[i]) * 0.5 * count_kernels;
        // initialitation of weight coefficients vector
        init_weight = (s_y[i] - Lasso_part)/(s_kernels_sum_sum + kernel_ridge_part);
        for (j = 0; j < count_kernels; j++)
            s_weights_vec_vec[i][j] = init_weight;
        
        for (j = 0; j < max_learning_cycles; j++) 
            {
            for (k = 1; k < count_geometries; k++)
                {
                // get the model prediction for another geometry
                s_y_temp_part_vec |= s_kernels_rows_sums_vec_vec[k];
                s_y_temp_part_vec *= s_weights_vec_vec[i];
                y_pred = s_y_temp_part_vec.sum();
                // calculating kernel ridge part of prediction
                if (kernel_ridge_part != 0)
                    y_pred_2 = y_pred * kernel_ridge_part/(sum_kernels_base - s_kernels_diff_sums_vec[k]);
                else
                    y_pred_2 = 0;
                
                y_pred_3 = y_pred + y_pred_2 + Lasso_part;
                // calculating difference
                y_diff = y_pred_3 - s_y[k + (i * count_geometries)];
                // calculating new weights
                s_y_temp_part_vec |= s_kernels_diff_vec_vec[k];
                s_y_temp_part_vec /= s_kernels_diff_sums_vec[k];
                // optimalization of kernel ridge regression
                if (kernel_ridge_part != 0)
                    {
                    s_y_temp_vec |= s_weights_vec_vec[i];
                    s_y_temp_vec /= s_weights_vec_vec[i].sum()/y_pred_2;
                    s_y_temp_part_vec -= s_y_temp_vec;
                    }
                if (element_invariant == true and kernel_classes_index != nullptr)
                    {
                    // for element ivariant kernels
                    for (l = 0; l < count_kernels; l++)
                        {
                        element_sum = 0;
                        element_count = 0;
                        for (m = 0; m < count_kernels; m++)
                            {
                            if (kernel_classes_index[m] == kernel_classes_index[l] and m < l)
                                break;
                            if (kernel_classes_index[m] == kernel_classes_index[l] and m >= l)
                                {
                                element_sum += s_y_temp_part[l];
                                element_count += 1; 
                                }
                            }
                        if (element_count > 1)
                            {
                            element_sum /= element_count;
                            s_y_temp_part[l] = element_sum;
                            for (m = l + 1; m < count_kernels; m++)
                                {
                                if (kernel_classes_index[m] == kernel_classes_index[l])
                                    s_y_temp_part[m] = element_sum;
                                }
                            }
                        }
                    }
                s_y_temp_part_vec /= (T(count_geometries)/(learning_rate * y_diff));
                s_weights_vec_vec[i] -= s_y_temp_part_vec;
                }
            // convergence criteria check
            if (max_Loss > 0)
                {
                Loss = Loss_Function(i);
                if (Loss < max_Loss)
                   break;
                }
            }
        }
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Save_fitted(const string weights)
    {
    unsigned int i, j;
    unsigned int count_kernels, count_levels;
    ofstream weightsfile(weights);
    string line;
    
    count_kernels = s_count_kernels;
    count_levels = s_count_levels;
    if (weightsfile.is_open())
        {
        for (i = 0; i < count_levels; i++)
            {
            for (j = 0; j < count_kernels; j++)
                {
                line = to_string(s_weights_vec_vec[i][j]);
                weightsfile << line << '\n';
                }
            }
        weightsfile.close();
        }
    else
        {
        laerror("not open file for weights");
        return(1);
        }
    
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Load_fitted(const string weights)
    {
    unsigned int i, j;
    unsigned int count_kernels, count_levels;
    ifstream weightsfile(weights);
    string line;
    T line_value;
    
    count_kernels = s_count_kernels;
    count_levels = s_count_levels;
    // initialization of REG_distances_mat and REG_kernels_mat if necessary
    if (s_distances_mat_init == false)
        {
        Build_s_distances_mat(s_count_geometries, s_geometries, s_distances);
        s_distances_mat_init = true;
        }
    if (s_descriptors_mat_init == false)
        {
        Build_s_descriptor_mat(s_count_geometries, s_distances, s_desc);
        s_descriptors_mat_init = true;
        }
    if (s_kernels_mat_init == false)
        {
        Build_s_kernels_mat(s_count_geometries, s_desc, s_kernels_vec_mat);
        Build_s_kernels_mat_sums();
        s_kernels_mat_init = true;
        }
    if (weightsfile.is_open())
        {
        for (i = 0; i < count_levels; i++)
            for (j = 0; j < count_kernels; j++)
                {
                if (getline(weightsfile, line))
                    {
                    line_value = stod(line);
                    s_weights_vec_vec[i][j] = line_value;
                    }
                else
                    {
                    laerror("not open file with weights");
                    return(1);
                    }
                }
        weightsfile.close();
        }
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Get_predictions(const unsigned int count_geometries, const unsigned int count_levels,
const T* geom, const unsigned int* surfaces, T* predictions)
    {
    unsigned int i, j, k;
    unsigned int count_kernels;
    T distance;
    T Lasso_part;
    T kernel_ridge_part;
    T sum_kernels;
    T pred_1, pred_2, pred_3;
    
    if (s_base_init == false)
        {
        laerror("regression model not inicialized");
        return(1);
        }
    for (i = 0; i < count_levels; i++)
        if (surfaces[i] >= s_count_levels)
            {
            laerror("invalid number of surface level");
            return(1);
            }
    
    count_kernels = s_count_kernels;
    
    for (i = 0; i < count_geometries; i++)
        {
        // building distances, descriptors and kernels matrices
        Build_s_distances_mat(1, geom + (i * s_count_dimensions * s_count_kernels), s_distances_pred);
        Build_s_descriptor_mat(1, s_distances_pred, s_desc_pred);
        Build_s_kernels_mat(1, s_desc_pred, s_kernels_pred_vec_mat);
        // replacing errored rsum() function
        for (j = 0; j < count_kernels; j++)
            {
            sum_kernels = 0;
            for (k = 0; k < count_kernels; k++)
                sum_kernels += s_kernels_pred[(j * count_kernels) + k];
            
            s_y_temp[j] = sum_kernels;
            }
        sum_kernels = s_y_temp_vec.sum();
        
        for (j = 0; j < count_levels; j++)
            {
            Lasso_part = s_lambda[surfaces[j]] * s_alpha[surfaces[j]];
            kernel_ridge_part = s_lambda[surfaces[j]] * (1 - s_alpha[surfaces[j]]) * 0.5 * count_kernels;
            // getting prediction
            if (surfaces[j] < s_count_levels)
                {
                s_y_temp_part_vec |= s_y_temp_vec;
                s_y_temp_part_vec *= s_weights_vec_vec[surfaces[j]];
                }
            else
                {
                laerror("invalid number of surface level");
                return(1);
                }
            
            pred_1 = s_y_temp_part_vec.sum();
            if (kernel_ridge_part != 0)
                {
                pred_2 = pred_1 * kernel_ridge_part/sum_kernels;
                }
            else
                pred_2 = 0;
            
            pred_3 = Lasso_part;
            predictions[(i * count_levels) + j] = pred_1 + pred_2 + pred_3;
            }
        }
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Score(const unsigned int count_geometries, const unsigned int count_levels,
const T* geom, const unsigned int* surfaces, T* predictions, T* y_values, T* score)
    {
    unsigned int i;
    T RMSE;
    
    s_Get_predictions(count_geometries, count_levels, geom, surfaces, predictions);
    for (i = 0; i < count_geometries * count_levels; i++)
        RMSE += (predictions[i] - y_values[i]) * (predictions[i] - y_values[i]);
    
    RMSE = RMSE/(count_geometries * count_levels);
    RMSE = std::sqrt(RMSE);
    score[0] = RMSE;
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Set_params(const string descriptor, const string kernel, const T* params)
    {
    unsigned int i;
    unsigned int count_levels;
    unsigned int new_descriptor_id;
    unsigned int new_kernel_id;
    
    if (s_base_init == false)
        {
        laerror("regression model not initialized");
        return(1);
        }
    
    count_levels = s_count_levels;
    s_kernel_type = kernel;
    if (kernel != "Gaussian" and kernel != "Laplacian" and kernel != "Exponential" and kernel != "Matern")
        {
        laerror("invalid type of kernel");
        return(1);
        }
    if (kernel == "Gaussian")
        new_kernel_id = 0;
    if (kernel == "Laplacian")
        new_kernel_id = 1;
    if (kernel == "Exponential")
        new_kernel_id = 2;
    if (kernel == "Matern")
        new_kernel_id = 3;
        
    s_descriptor_type = descriptor;
    if (descriptor == "RE")
        new_descriptor_id = 0;
    if (descriptor == "Coulomb_matrix")
        new_descriptor_id = 1;
    if (descriptor == "Inverse_pairs")
        new_descriptor_id = 2;
    
    if (new_kernel_id != s_kernel_id)
        {
        s_kernel_id = new_kernel_id;
        s_kernels_mat_init = false;
        }
    if (new_descriptor_id != s_descriptor_id)
        {
        s_descriptor_id = new_descriptor_id;
        s_kernels_mat_init = false;
        s_descriptors_mat_init = false;
        }
    
    // copy parameters and data
    for (i = 0; i < count_levels; i++)
        s_lambda[i] = params[i];
    
    for (i = count_levels; i < (2 * count_levels); i++)
        {
        s_alpha[i - count_levels] = params[i];
        if (s_alpha[i - count_levels] > 1)
            s_alpha[i - count_levels] = 1;
        if (s_alpha[i - count_levels] < 0)
            s_alpha[i - count_levels] = 0;
        }
    s_sigma = params[2 * count_levels];
    
    if (s_kernel_id == 3)
        for (i = 0; i < ((2 * count_levels) + 2); i++)
            s_params[i] = params[i];
    else
        for (i = 0; i < ((2 * count_levels) + 1); i++)
            s_params[i] = params[i];
    
    if (s_kernel_id == 3)
        {
        s_n = static_cast<unsigned int>(params[(2 * count_levels) + 1] + 0.5);
        if (s_Matern_comb_number != nullptr)
            {
            delete[] s_Matern_comb_number;
            s_Matern_comb_number = nullptr;
            }
            {
            s_Matern_comb_number = new T[s_n];
	    if(!s_Matern_comb_number) laerror("allocation error of s_Matern_comb_number");
            Matern_coefficient(s_n, s_Matern_comb_number);
            }
        }
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Get_params(string* descriptor, string* kernel, T* params)
    {
    unsigned int i;
    unsigned int count_levels;
    
    if (s_base_init == false)
        {
        laerror("regression model not initialized");
        return(1);
        }
    
    count_levels = s_count_levels;
    descriptor[0] = s_descriptor_type;
    kernel[0] = s_kernel_type;
    
    if (s_kernel_id != 3)
        for (i = 0; i < ((2 * count_levels) + 1); i++)
            params[i] = s_params[i];
    else
        for (i = 0; i < ((2 * count_levels) + 2); i++)
            params[i] = s_params[i];
    
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>:: s_Set_extern_descriptor(unsigned int (*extern_func)(const unsigned int, const unsigned int, const T*, T*))
    {
    extern_desc = extern_func;
    extern_descriptor = true;
    s_descriptor_id = 3;
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Get_weight_coeff(const unsigned int s_number, T* weights)
    {
    unsigned int i;
    unsigned int count_kernels;
    
    count_kernels = s_count_kernels;
    for (i = 0; i < count_kernels; i++)
        weights[i] = s_weights_vec_vec[s_number][i];
    
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Load_ML_NEA_geometries(const unsigned int count_geometries, const T* geometries)
    {
    ML_NEA_count_geometries = count_geometries;
    s_ML_NEA_geometries_mat = NRMat<T>(geometries, s_count_dimensions * s_count_kernels, count_geometries);
    s_ML_NEA_geometries = s_ML_NEA_geometries_mat.operator T*();
    s_ML_NEA_input_init = true;
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Load_ML_NEA_geometries(const unsigned int count_geometries, const string geometries_file)
    {
    unsigned int i, j, k, l;
    unsigned int count_dimensions, count_kernels;
    unsigned int line_size, numbers_size;
    ifstream geomfile(geometries_file);
    string line;
    T line_value;
    const string numbers = "+-0123456789.,";
    string readed_number;
    bool read_switch;
    bool decimal_switch;
    
    count_dimensions = s_count_dimensions;
    count_kernels = s_count_kernels;
    ML_NEA_count_geometries = count_geometries;
    s_ML_NEA_geometries_mat = NRMat<T>(s_count_dimensions * s_count_kernels, count_geometries);
    s_ML_NEA_geometries = s_ML_NEA_geometries_mat.operator T*();
    
    l = 0;
    numbers_size = numbers.size();
    if (geomfile.is_open())
        {
        for (i = 0; i < 3 * count_dimensions * count_kernels * count_geometries; i++)
            {
            if (getline(geomfile, line))
                {
                read_switch = false;
                decimal_switch = false;
                readed_number = "";
                line_size = line.size();
                for (j = 0; j < line_size; j++)
                    {
                    read_switch = false;
                    for (k = 0; k < numbers_size; k++)
                        {
                        if (line[j] == numbers[k])
                            {
                            readed_number.append(line, j, 1);
                            read_switch = true;
                            if (line[j] == numbers[12] or line[j] == numbers[13])
                                decimal_switch = true;
                            }
                        }
                    if (read_switch == false)
                        {
                        if (decimal_switch == true and readed_number != "")
                            {
                            line_value = stod(readed_number);
                            s_ML_NEA_geometries[l] = line_value;
                            l++;
                            }
                        readed_number = "";
                        read_switch = false;
                        decimal_switch = false;
                        }
                    }
                if (decimal_switch == true and readed_number != "")
                    {
                    line_value = stod(readed_number);
                    s_ML_NEA_geometries[l] = line_value;
                    l++;
                    }
                }
            if (l == count_dimensions * count_kernels * count_geometries)
                break;
            }
        geomfile.close();
        }
    else
        {
        laerror("file with geometry is not open");
        return(1);
        }
        
    if (l < count_dimensions * count_kernels * count_geometries)
        {
        laerror("input ML-NEA geometries not complete");
        return(1);
        }
    
    s_ML_NEA_input_init = true;
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Predict_ML_NEA_geometries()
    {
    unsigned int i;
    T control_value;
    // getting predistions
    if (s_base_init == false)
        {
        laerror("regression model not initialized");
        return(1);
        }
    if (s_ML_NEA_input_init == false)
        {
        laerror("ML-NEA model input data not initialized");
        return(1);
        }
    
    s_ML_NEA_surf_numbers_vec = NRVec<unsigned int>(s_count_levels);
    s_ML_NEA_surf_numbers = s_ML_NEA_surf_numbers_vec.operator unsigned int*();
    for (i = 0; i < s_count_levels; i++)
        s_ML_NEA_surf_numbers[i] = i;
    
    s_ML_NEA_y_mat = NRMat<T>(ML_NEA_count_geometries, s_count_levels);
    s_ML_NEA_y = s_ML_NEA_y_mat.operator T*();
    control_value = s_Get_predictions(ML_NEA_count_geometries, s_count_levels, s_ML_NEA_geometries, s_ML_NEA_surf_numbers, s_ML_NEA_y);
    if (control_value == 0)
        s_ML_NEA_surf_init = true;
    return(control_value);
    }


template <typename T>
unsigned int REGSURF<T>::s_Compute_ML_NEA_spectra(const T E_min, const T step, const unsigned int count_steps, const T delta)
    { // sigma(E) = (e^2 *h)/(4*m_e*c*E_0*E) * sum_{n=1}^N\(1/Np * sum_{i=1}^Np * delta_E_0n(x_i) * f_0n(x_i) * f_g(E - delta_E_0n, delta))
      // delta_E_0n(x_i) = delta_E_min_n/delta_E_0n
      // f_g = 1/(sqrt(Phi) * delta * delta) * exp(-((E-delta_E_0n)^2)/(0.5 * delta * delta))
    unsigned int i, j, k;
    unsigned int count_kernels, count_levels, count_geometries;
    T y_min;
    T h = 6.62607015E-34;
    T e = 1.602176634E-19;
    T m_e = 9.1093837E-31;
    T c = 299796000;
    T E_0 = 8.8541878128E-12;
    T Phi = (1.00 + std::sqrt(5))/2.00;
    T pre_sum_part;
    T E;
    T delta_E_0n;
    T f_0n;
    T f_g;
    T sigma_part;

    count_kernels = s_count_kernels;
    count_levels = s_count_levels;
    count_geometries = ML_NEA_count_geometries;
    if (s_base_init == false)
        {
        laerror("regression model not initialized");
        return(1);
        }
    if (s_ML_NEA_input_init == false)
        {
        laerror("ML-NEA model input data not initialized");
        return(1);
        }
    
    // calculating minima of surfaces
    s_ML_NEA_min_y_vec = NRVec<T>(s_count_levels);
    s_ML_NEA_min_y = s_ML_NEA_min_y_vec.operator T*();
    for (i = 0; i < count_levels; i++)
        {
        y_min = s_ML_NEA_y[i * count_kernels];
        for (j = 1; j < count_kernels; j++)
            {
            if (s_ML_NEA_y[(i * count_kernels) + j] < y_min)
                y_min = s_ML_NEA_y[(i * count_kernels) + j];
            }
        s_ML_NEA_min_y[i] = y_min;
        }
    // calculating spectra
    s_ML_NEA_spectra_vec = NRVec<T>(count_steps);
    s_ML_NEA_spectra = s_ML_NEA_spectra_vec.operator T*();
    pre_sum_part = (e * e * h)/(4 * m_e * c * E_0);
    E = E_min;
    for (i = 0; i < count_steps; i++)
        s_ML_NEA_spectra[i] = 0;
    
    for (i = 0; i < count_geometries; i++)
        {
        for (j = 0; j < count_levels; j++)
            {
            // calculating the oscilator strenght
            delta_E_0n = s_ML_NEA_y[(j * count_geometries) + i];
            f_0n = s_ML_NEA_min_y[j]/delta_E_0n;
            E = E_min;
            if (delta_E_0n > 0 and f_0n > 0)
                for (k = 0; k < count_steps; k++)
                    {
                    // calculating gaussian function f_g partial intensity and adding to ML_NEA intensities vector
                    if (abs(E - delta_E_0n)/delta < 100)
                        {
                        f_g = 1/(std::sqrt(Phi) * delta);
                        f_g *= std::exp(-((E - delta_E_0n)*(E - delta_E_0n))/(0.5 * delta * delta));
                        if (E > 0 and f_g > 0)
                            {
                            sigma_part = pre_sum_part/E * delta_E_0n * f_0n * f_g;
                            s_ML_NEA_spectra[k] += sigma_part;
                            }
                        }
                    E += step;
                    }
            }
        }
    s_ML_NEA_spectra_init = true;
    return(0);
    }


template <typename T>
unsigned int REGSURF<T>::s_Save_ML_NEA_spectra(const string ML_NEA_spectra_file)
    {
    unsigned int i;
    unsigned int spectra_size;
    T spectra_sum;
    ofstream ML_NEA_file(ML_NEA_spectra_file);
    string line;
    
    spectra_size = s_ML_NEA_spectra_vec.size();
    spectra_sum = s_ML_NEA_spectra_vec.sum();
    if (ML_NEA_file.is_open())
        {
        for (i = 0; i < spectra_size; i++)
            {
            line = to_string(s_ML_NEA_spectra[i]/spectra_sum);
            ML_NEA_file << line << '\n';
            }
        ML_NEA_file.close();
        }
    else
        {
        laerror("file for ML-NEA spectra not open");
        return(1);
        }
    
    return(0);
    }


template <typename T>
REGSURF<T>::~REGSURF()
    {
    // dealocating memory to avoid memory leaks
    if (s_lambda != nullptr)
        delete[] s_lambda;
    if (s_alpha != nullptr)
        delete[] s_alpha;
    if (s_params != nullptr)
        delete[] s_params;
    if (s_Matern_comb_number != nullptr)
        delete[] s_Matern_comb_number;
    if (s_y_min != nullptr)
        delete[] s_y_min;
    if (s_geom_file != nullptr)
        delete[] s_geom_file;
    if (s_y_file != nullptr)
        delete[] s_y_file;
    if (s_Z_file != nullptr)
        delete[] s_Z_file;
    }
} // end of namespace

using namespace LA;

//enforce instantiation


template class REGSURF<double>;
