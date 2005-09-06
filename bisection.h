#ifndef _BISECTION_H
#define _BISECTION_H
//general bisection search
//returns dm-1 on failure, otherwise number between dm and hm

template<typename INDEX, typename COMPAR, typename SUBJECT>
INDEX bisection_find(INDEX dm, INDEX hm, const SUBJECT *key, const SUBJECT *base , unsigned int lead_dimension_base, COMPAR (*cmp)(const SUBJECT *, const SUBJECT *)) 
{
if(dm>hm) return(dm-1);
if(dm==hm) return  (*cmp)(base+dm*lead_dimension_base,key)? dm-1 :dm;
INDEX sm;
INDEX dm0=dm;
--dm;
++hm;
do
          {
          sm = (dm+hm)/2;
	  COMPAR q = (*cmp)(base+sm*lead_dimension_base,key);
          if (!q) return(sm);
          else if (q<0) dm = sm; else hm = sm;
          }
while (hm > dm+1);
return(dm0-1);
}

#endif
