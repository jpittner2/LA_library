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

using namespace std;
using namespace LA;


int main(int argc, char *argv[])
{
// unsigned int i;

double x;
double y;
bool end = false;
string user_input;
string model_x = "reg_test_x.csv";
string model_y = "reg_test_y.csv";
string model_p = "reg_test_p.csv";
REG<double> reg(model_x, model_y, model_p);
/*
double* reference_value;
double* score;
double RMSE;
double Loss;
double* params;
double* weights;
string kernel;

reference_value = new double[1];
score = new double[1];
params = new double[4];
weights = new double[451];

reference_value[0] = 1.00;
*/
// creating model
// fitting model
reg.REG_Fit(0.0, 20);
//reg.REG_Fit_grad(0.1, 50, 0.2);
// get predictions

cout << "Program for testing regression model \n" << endl;
/*
reg.REG_Get_weight_coeff(weights);
cout << "The weight coefficients are: " << kernel << endl;
for (i = 0; i < 451; i++)
    cout << weights[i] << endl;
*/
while (not end)
    {
    cout << "Please enter the x value or y to end \n" << endl;
    cin >> user_input;
    if (user_input == "y")
        {
        end = true;
        break;
        }
    x = stod(user_input);
    
    reg.REG_Get_predictions(&x, &y, 1);
    cout << "The model prediction is: " << y << endl;
    /*
    reg.REG_Get_Score(&x, &y, 1, reference_value, score);
    cout << "The model prediction is: " << y << endl;
    cout << "The reference value is: " << reference_value[0] << endl;
    cout << "The score is: " << score[0] << endl;
    
    RMSE = reg.REG_RMSE();
    Loss = reg.Loss_Function();
    cout << "The RMSE is: " << RMSE << endl;
    cout << "The Loss is: " << Loss << endl;
    
    reg.REG_Get_params(params, &kernel);
    cout << "The parameters of model are: " << endl;
    for (i = 0; i < 4; i++)
        cout << "param: " << params[i] << endl;
        
    cout << "The type of kernel is: " << kernel << endl;
    cout << endl;
    
    params[0] = 0;
    kernel = "Gaussian";
    reg.REG_Set_params(params, kernel);
    
    reg.REG_Get_params(params, &kernel);
    cout << "The new parameters o model are: " << RMSE << endl;
    for (i = 0; i < 4; i++)
        cout << "param: " << params[i] << endl;
        
    cout << "The new type of kernel is: " << kernel << endl;
    cout << endl;
    */
    }
/*reg.REG_Save_fitted("reg_test_m.csv");
reg.REG_Load_fitted("reg_test_m.csv");
end = false;
while (not end)
    {
    cout << "Please enter the x value or y to end \n" << endl;
    cin >> user_input;
    if (user_input == "y")
        {
        end = true;
        break;
        }
    x = stod(user_input);
    reg.REG_Get_predictions(&x, &y, 1);
    cout << "The model prediction is: " << y << endl;
    }*/
return(0);
};
