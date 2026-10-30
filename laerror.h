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
#include <string>

#include <signal.h>

namespace LA {

extern "C" {
extern int traceback(int flags);
extern void sigtraceback(int sig,int flags);
}

extern const bool _LA_warn_empty_copyonwrite;
extern  bool _LA_count_check; //intentionally not const


//exception class for laerror
class LAerror : public std::exception
	{
	public:
	std::string msg; //own copy: the argument may be a local buffer of an error handler
	LAerror(const char *s) : msg(s?s:"unspecified") {};
	const char *what() const noexcept {return msg.c_str();};
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

extern int endianity();

extern bool LA_error_list_files;

}//namespace

#endif
