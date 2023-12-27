#include <stdlib.h>
#include "la_random.h"

namespace LA {

WEAK_SYMBOL  double randdouble() {return random()/(1.+RAND_MAX);}
WEAK_SYMBOL  double randdoublesigned() {return 2.*random()/(1.+RAND_MAX)-1.;}
WEAK_SYMBOL  int randint32() {return random();}
WEAK_SYMBOL  uint64_t randint64() {uint64_t r = random(); r<<=32; r|= random(); return r; }
}//namespace

