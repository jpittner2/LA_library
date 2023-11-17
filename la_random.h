#ifndef _LA_RANDOM_H
#define _LA_RANDOM_H

namespace LA {

//RANDOM numbers defaulting to standard library but switchable to user's functions
#ifndef RANDDOUBLE
#define RANDDOUBLE randdouble
#endif

#ifndef RANDDOUBLESIGNED
#define RANDDOUBLESIGNED randdoublesigned
#endif

#ifndef RANDINT32
#define RANDINT32 randint32
#endif

extern "C" {
extern double randdouble();
extern double randdoublesigned();
extern int randint32();
}

#ifdef __GNUC__
#define WEAK_SYMBOL __attribute__((weak)) 
#else
#define WEAK_SYMBOL 
#endif

}//namespace

#endif
