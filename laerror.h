/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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
#ifndef _LAERROR_H_
#define _LAERROR_H_
#include <iostream>

namespace LA {

//exception class for laerror
class LAerror
	{
	public:
	const char *msg;
	LAerror(const char *s) {msg=s;};
	};

#ifdef __GNUG__
	#define laerror(X) LA::laerror2(X, __PRETTY_FUNCTION__)
#else
	#define laerror(X) LA::laerror2(X, __func__)
#endif

extern void laerror2(const char *, const char *);

inline std::ostream & operator<<(std::ostream &s, const LAerror &x) {
s << x.msg;
return s;
}

}//namespace

#endif
