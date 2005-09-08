template<typename T>
const NRVec<T> inversepermutation(const NRVec<T> &p, const T offset=0)
{
int n=p.size();
NRVec<T> q(n);
if(!offset) for(int i=0; i<n; ++i) q[p[i]]=i;
else for(int i=0; i<n; ++i) q[p[i]-offset]=i+offset;
return q;
}
