#include <stdlib.h>
#include "la_random.h"

namespace LA {

extern "C" {
WEAK_SYMBOL  double randdouble() {return random()/(1.+RAND_MAX);}
WEAK_SYMBOL  double randdoublesigned() {return 2.*random()/(1.+RAND_MAX)-1.;}
WEAK_SYMBOL  int randint32() {return random();}
}
}//namespace

