// LA errorr handler
#include <iostream>
#include "laerror.h"
void laerror(const char *s1)
{
  std::cerr << "LA:ERROR - ";
  if(!s1)
    std::cerr << "udefined.\n";
  else {
    if(s1) std::cerr << s1;
	std::cerr << "\n";
  }
  throw LAerror(s1);
  exit(1);
}

