#ifndef _BISECTION_H
#define _BISECTION_H
//general bisection search
//returns dm-1 on failure, otherwise number between dm and hm
//cmp returns 0 on equal, >0 if first > second argument

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


// interpolation search
//distance is +-0, on its quality the convergence speed depends

template<typename INDEX, typename DISTANCE, typename SUBJECT>
int interpolation_find(INDEX dm0, INDEX high, const SUBJECT *x, const SUBJECT *base , unsigned int lead_dimension_base, DISTANCE (*dist)(const SUBJECT *, const SUBJECT *))
{
        INDEX low = dm0;
        INDEX mid;
	DISTANCE d1,d2,d3,d4;

        while(1)
		{
		d1=(*dist)(x,base+low*lead_dimension_base);
		d2=(*dist)(base+high*lead_dimension_base,x);
		cout << "intlp "<<low<<" "<<high<<" "<<d1<<" "<<d2<,endl
		if(d1<=0 || d2 >=0)  break;
		d3=(*dist)(base+high*lead_dimension_base,base+low*lead_dimension_base);
	        mid = low + (int)((d1 * (high-low)) / d3+.5); 
		d4=(*dist)(x,base+mid*lead_dimension_base);
		cout << "intlp2 "<<mid<<" "<<d3<<" "<<d4<<endl;
                if(d4>0)
                    low = mid + 1;
                else if(d3<0)
                    high = mid - 1;
                else
                    return mid;
        }  

	if (d1==0) return low;
	if (d2==0) return high;
	return dm0-1; // Not found
}



#endif
