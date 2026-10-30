#include <stdlib.h>
#include "la_random.h"

namespace LA {

WEAK_SYMBOL  double randdouble() {return random()/(1.+RAND_MAX);}
WEAK_SYMBOL  double randdoublesigned() {return 2.*random()/(1.+RAND_MAX)-1.;}
WEAK_SYMBOL  unsigned int randint32() {unsigned int r = random(); r<<=1; r^= random(); return r;}
WEAK_SYMBOL  uint64_t randint64() {uint64_t r = random(); r<<=31; r|= random(); r<<=2; r|= random()&3; return r; } //random() gives 31 bits
}//namespace

