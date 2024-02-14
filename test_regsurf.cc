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

using namespace std;
using namespace LA;



int main(int argc, char *argv[])
{
unsigned int i, j;
unsigned int count_levels;
unsigned int count_kernels;
double x;
double y;
double RMSE;
double Loss;
bool end = false;
string user_input;
string model_geoms = "xyz.dat";
string model_y_values = "E_sum.dat";
string model_params = "conf_surf.dat";
string model_Z_file = "Z.dat";
string model_test_geom = "nea_geoms.xyz";
/*
double* weight_coeff;
double* params;
string model_weight_file;
string* descriptor;
string* kernel;
*/
count_levels = 10;
count_kernels = 12;/*
descriptor = new string[1];
kernel = new string[1];

weight_coeff = new double[count_kernels * count_levels];
params = new double[22];
model_weight_file = "weights.dat";
*/
REGSURF<double> regsurf(model_geoms, model_y_values, model_params, model_Z_file);
// creating model
// fitting model
regsurf.s_Fit(0.0, 100, 0.5);
//reg.REG_Fit_grad(0.1, 50, 0.2);

// get predictions

cout << "Program for testing regression model \n" << endl;
cout << "The root mean square errors: " << endl;
for (i = 0; i < count_levels; i++)
    {
    RMSE = regsurf.s_RMSE(i);
    cout << RMSE << endl;
    }
cout << "The values of Loss_function: " << endl;
for (i = 0; i < count_levels; i++)
    {
    Loss = regsurf.Loss_Function(i);
    cout << Loss << endl;
    }
/*
cout << "The weights of kernels: " << endl;
for (i = 0; i < count_levels; i++)
    {
    cout << "Level " << i << ": ";
    regsurf.s_Get_weight_coeff(i, weight_coeff + (i * count_kernels));
    for (j = 0; j < count_kernels; j++)
        cout << weight_coeff[(i * count_kernels) + j] << endl;
    }

regsurf.s_Save_fitted(model_weight_file);
regsurf.s_Load_fitted(model_weight_file);
regsurf.s_Get_params(descriptor, kernel, params);

cout << "The descriptor is: " << descriptor[0] << endl;
cout << "The kernel is: " << kernel[0] << endl;
cout << "The parameters of model are: ";
for (i = 0; i < (2 * count_levels) + 1; i++)
    cout << params[i] << endl;

descriptor[0] = "Inverse_pairs";
kernel[0] = "Laplacian";
params[1] = 0.2;
regsurf.s_Set_params(descriptor[0], kernel[0], params);

regsurf.s_Get_params(descriptor, kernel, params);
cout << "The new descriptor is: " << descriptor[0] << endl;
cout << "The new kernel is: " << kernel[0] << endl;
cout << "The new parameters of model are: ";
for (i = 0; i < (2 * count_levels) + 1; i++)
    cout << params[i] << endl;
*/
regsurf.s_Load_ML_NEA_geometries(50000, "nea_geoms.xyz");
regsurf.s_Predict_ML_NEA_geometries();
regsurf.s_Compute_ML_NEA_spectra(0.00, 0.0025, 4000, 0.2);
regsurf.s_Save_ML_NEA_spectra("spectra.dat");
return(0);
};
