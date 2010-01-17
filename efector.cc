/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>
    This file contributed by <miroslav.sulc@jh-inst.cas.cz>


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

#include "la_traits.h"
#include <iostream>
#include <sstream>

namespace LA {

//--------------------------------------------------------------------------------
class PrintNRCMat{
private:
public:
	std::ostringstream s;
	PrintNRCMat(const PrintNRCMat &_fw);
	PrintNRCMat(const NRCMat& _K, const int _prec = 5, const int _pl = 0, const int _pt = 0, const double _tol = 1.0e-12);
	friend ostream& operator<<(ostream& os, const PrintNRCMat& fw){
		return (os << fw.s.str());
	}
};
//--------------------------------------------------------------------------------
PrintNRCMat::PrintNRCMat(const PrintNRCMat &_fw){
	this->s.str(_fw.s.str());
}
//--------------------------------------------------------------------------------
PrintNRCMat::PrintNRCMat(const NRCMat &_K, const int _prec, const int _pl, const int _pt, const double _tol){
/*
	formatuje realnou cast komplexni matice _K ve forme tabulky, napr.
			----------------------------------------------
			| +6.80375e+00 | +5.66198e+00 | +8.23295e+00 |
			----------------------------------------------
			| -3.29554e+00 | -4.44451e+00 | -4.52059e-01 |
			----------------------------------------------
			| -2.70431e+00 | +9.04459e+00 | +2.71423e+00 |
			----------------------------------------------

	parametry:
		_prec	presnost pro cout << setprecision(...)
		_pl		left padding - odsazeni tabulky zleva
		_pt		top padding - odsazeni tabulky seshora
		_tol	cisla mensi v abs. hodnote nez _tol se zobrazi jako 0
*/
	const int n = _K.nrows();
	const int m = _K.ncols();
	double val(0.0);


	const int w = _prec + 7;
	const int sirka = m*(w+3) + 1;
	const int pl = (_pl>0&&_pl<10)?_pl:0;
	const int pt = (_pt>0&&_pt<10)?_pt:0;

	const string vl = string(pl, ' ');
	const string radek = string(sirka, '-');
	const string vypln = string(w, ' ');

	this->s << scientific << setprecision(_prec) << showpos;
	this->s << string(pt, '\n');
	const string name = _K.getname();
	if(name != ""){
		this->s << vl << "matrix '" << name << "': " << endl;
	}
	this->s << vl << radek << endl;
	for(register int i = 0;i < n;i++){
		this->s << vl;
		for(register int j = 0;j < m; j++){
			val = (_K[i][j]).real();

			this->s << setw(2) << right << "| " << setw(w) << left;
			if(std::abs(val)<_tol){
				this->s << val;
			}else{
				this->s << val;
			}
			this->s << setw(1) << " ";
		}
		this->s << "|" << endl;
		this->s << vl << radek << endl;
	}
}


}//namespace
