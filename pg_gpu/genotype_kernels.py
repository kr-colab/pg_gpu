"""
Fused CUDA kernels for genotype-based LD statistics (DD, Dz, pi2).

All multi-population pi2 and Dz cases use RawKernels that compute one
output per GPU thread with all arithmetic in registers. Genotype counts
use 9-way layout: (g1..g9) per locus pair per population, following the
moments convention.

DD uses the ld_statistics_genotype polynomial formulas directly (few
calls, already fast, not worth fused kernels).
"""
import cupy as cp


def _launch(kernel, args, M):
    """Launch a fused kernel with M output elements."""
    kernel(((int(M) + 255) // 256,), (256,), args)


# ---------------------------------------------------------------------------
# DD kernels
# ---------------------------------------------------------------------------
# Genotype DD does not have fused CUDA kernels. The number of DD calls is
# small (at most 10 for 4 populations), so we dispatch directly to the
# ld_statistics_genotype formula functions (dd_geno_single, dd_geno_between)
# in the compute_all_dd_geno dispatch function below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dz kernels
# ---------------------------------------------------------------------------

_DZ_PRECOMP_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*g, double*ns, double*D, double*A, double*Bv,
       double*Wjj, double*Ucoef, double*Vcoef, const long long N){
    long long idx=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    if(idx>=N)return;
    const double*row=g+9*idx;
    double n1=row[0],n2=row[1],n3=row[2],n4=row[3],n5=row[4],n6=row[5],n7=row[6],n8=row[7],n9=row[8];
    double n=n1+n2+n3+n4+n5+n6+n7+n8+n9;
    double x00=n1+.5*n2+.5*n4+.25*n5, x01=.5*n2+n3+.25*n5+.5*n6;
    double x10=.5*n4+.25*n5+n7+.5*n8, x11=.25*n5+.5*n6+.5*n8+n9;
    double d=x00*x11-x01*x10;
    double a=-n1+n3-n4+n6-n7+n9;
    double b=-n1-n2-n3+n7+n8+n9;
    double c=-n1+n3+n7-n9;
    double MA=2.*n2*n7+4.*n3*n7+2.*n5*n7+4.*n6*n7-2.*n1*n8+2.*n3*n8-2.*n4*n8+2.*n6*n8-4.*n1*n9-2.*n2*n9-4.*n4*n9-2.*n5*n9;
    double MB=2.*n3*n4+2.*n3*n5-2.*n1*n6-2.*n2*n6+4.*n3*n7+2.*n6*n7+4.*n3*n8+2.*n6*n8-4.*n1*n9-4.*n2*n9-2.*n4*n9-2.*n5*n9;
    ns[idx]=n; D[idx]=d; A[idx]=a; Bv[idx]=b;
    Wjj[idx]=c+a*b;
    Ucoef[idx]=d*(1.+b)+.25*MA;
    Vcoef[idx]=d*(1.+a)+.25*MB;
}''', "k", options=("-std=c++11",))

_DZ_DISTINCT_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*ns,const double*D,const double*A,const double*Bv,
       const int*I,const int*J,const int*K,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],j=J[t],k=K[t];
    out[t]=2.*D[i]*Bv[j]*A[k]/(ns[i]*(ns[i]-1.)*ns[j]*ns[k]);
}''', "k", options=("-std=c++11",))

_DZ_IIJ_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*ns,const double*Ucoef,const double*A,
       const int*I,const int*J,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],j=J[t];
    out[t]=2.*Ucoef[i]*A[j]/(ns[j]*ns[i]*(ns[i]-1.)*(ns[i]-2.));
}''', "k", options=("-std=c++11",))

_DZ_IJI_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*ns,const double*Vcoef,const double*Bv,
       const int*I,const int*J,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],j=J[t];
    out[t]=2.*Vcoef[i]*Bv[j]/(ns[j]*ns[i]*(ns[i]-1.)*(ns[i]-2.));
}''', "k", options=("-std=c++11",))

_DZ_IJJ_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*ns,const double*D,const double*Wjj,
       const int*I,const int*J,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],j=J[t];
    out[t]=2.*D[i]*Wjj[j]/(ns[j]*(ns[j]-1.)*ns[i]*(ns[i]-1.));
}''', "k", options=("-std=c++11",))

_DZ_III_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*g, double*out, const long long N){
    long long idx=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    if(idx>=N)return;
    const double*row=g+9*idx;
    double n1=row[0],n2=row[1],n3=row[2],n4=row[3],n5=row[4],n6=row[5],n7=row[6],n8=row[7],n9=row[8];
    double ns=n1+n2+n3+n4+n5+n6+n7+n8+n9;
    if(ns<4.){out[idx]=0.;return;}
    double x00=n1+.5*n2+.5*n4+.25*n5, x01=.5*n2+n3+.25*n5+.5*n6;
    double x10=.5*n4+.25*n5+n7+.5*n8, x11=.25*n5+.5*n6+.5*n8+n9;
    double D=x00*x11-x01*x10;
    double A=-n1+n3-n4+n6-n7+n9;
    double B=-n1-n2-n3+n7+n8+n9;
    double quartic=
        -(n2*n4)+3.*n1*n2*n4+n2*n2*n4+2.*n3*n4+4.*n1*n3*n4-n2*n3*n4
        -4.*n3*n3*n4+n2*n4*n4+2.*n3*n4*n4+2.*n1*n5-3.*n1*n1*n5-n1*n2*n5
        +2.*n3*n5+2.*n1*n3*n5-n2*n3*n5-3.*n3*n3*n5-n1*n4*n5+n3*n4*n5
        +2.*n1*n6-4.*n1*n1*n6-n2*n6-n1*n2*n6+n2*n2*n6+4.*n1*n3*n6
        +3.*n2*n3*n6-2.*n1*n4*n6-2.*n2*n4*n6-2.*n3*n4*n6+n1*n5*n6-n3*n5*n6
        +2.*n1*n6*n6+n2*n6*n6+2.*n2*n7+4.*n1*n2*n7+2.*n2*n2*n7+8.*n3*n7
        +4.*n1*n3*n7-4.*n3*n3*n7-n2*n4*n7+2.*n5*n7+2.*n1*n5*n7+n2*n5*n7
        +2.*n3*n5*n7-n4*n5*n7+2.*n6*n7-n2*n6*n7-2.*n4*n6*n7+n5*n6*n7
        +2.*n6*n6*n7-4.*n2*n7*n7-4.*n3*n7*n7-3.*n5*n7*n7-4.*n6*n7*n7
        +2.*n1*n8-4.*n1*n1*n8-2.*n1*n2*n8+2.*n3*n8-2.*n2*n3*n8-4.*n3*n3*n8
        -n4*n8-n1*n4*n8-2.*n2*n4*n8-n3*n4*n8+n4*n4*n8+n1*n5*n8+n3*n5*n8
        -n6*n8-n1*n6*n8-2.*n2*n6*n8-n3*n6*n8-2.*n4*n6*n8+n6*n6*n8
        +4.*n1*n7*n8-2.*n2*n7*n8+3.*n4*n7*n8-n5*n7*n8-n6*n7*n8
        +2.*n1*n8*n8+2.*n3*n8*n8+n4*n8*n8+n6*n8*n8
        +8.*n1*n9-4.*n1*n1*n9+2.*n2*n9+2.*n2*n2*n9+4.*n1*n3*n9+4.*n2*n3*n9
        +2.*n4*n9-n2*n4*n9+2.*n4*n4*n9+2.*n5*n9+2.*n1*n5*n9+n2*n5*n9
        +2.*n3*n5*n9+n4*n5*n9-n2*n6*n9-2.*n4*n6*n9-n5*n6*n9
        +4.*n1*n7*n9+4.*n3*n7*n9+4.*n4*n7*n9+2.*n5*n7*n9+4.*n6*n7*n9
        -2.*n2*n8*n9+4.*n3*n8*n9-n4*n8*n9-n5*n8*n9+3.*n6*n8*n9
        -4.*n1*n9*n9-4.*n2*n9*n9-4.*n4*n9*n9-3.*n5*n9*n9;
    double numer=.25*quartic+A*B*D;
    out[idx]=2.*numer/(ns*(ns-1.)*(ns-2.)*(ns-3.));
}''', "k", options=("-std=c++11",))


# ---------------------------------------------------------------------------
# pi2 kernels
# ---------------------------------------------------------------------------

_PI2_ALLDIFF_KERN = cp.RawKernel(r'''
extern "C" __global__ void k(const double*p,const double*r,const double*q,const double*s,
    const int*I,const int*J,const int*K,const int*L,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    out[t]=0.25*(p[I[t]]*r[J[t]]+r[I[t]]*p[J[t]])*(q[K[t]]*s[L[t]]+s[K[t]]*q[L[t]]);
}''', "k", options=("-std=c++11",))

_PI2_IIKK_KERN = cp.RawKernel(r'''
extern "C" __global__ void k(const double*BL,const double*BR,
    const int*I,const int*K,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    out[t]=BL[I[t]]*BR[K[t]];
}''', "k", options=("-std=c++11",))

_PI2_IIKL_KERN = cp.RawKernel(r'''
extern "C" __global__ void k(const double*BL,const double*q,const double*s,
    const int*I,const int*K,const int*L,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],k=K[t],l=L[t];
    out[t]=BL[i]*0.5*(q[k]*s[l]+s[k]*q[l]);
}''', "k", options=("-std=c++11",))

_PI2_IJKK_KERN = cp.RawKernel(r'''
extern "C" __global__ void k(const double*p,const double*r,const double*BR,
    const int*I,const int*J,const int*K,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],j=J[t],k=K[t];
    out[t]=0.5*(p[i]*r[j]+r[i]*p[j])*BR[k];
}''', "k", options=("-std=c++11",))

_PI2_SHARED_KERN = cp.RawKernel(r'''
extern "C" __global__ void k(const double*p,const double*r,const double*q,const double*s,
    const double*C00,const double*C01,const double*C10,const double*C11,
    const int*I,const int*J,const int*K,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],j=J[t],k=K[t];
    double sk=s[k],qk=q[k];
    out[t]=r[j]*(C00[i]*sk+C01[i]*qk)+p[j]*(C10[i]*sk+C11[i]*qk);
}''', "k", options=("-std=c++11",))

_PI2_IIIJ_KERN = cp.RawKernel(r'''
extern "C" __global__ void k(const double*AQ3,const double*AS3,
    const double*Q,const double*S,const double*nn,
    const int*I,const int*J,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],j=J[t];
    double ni=nn[i],nj=nn[j];
    double d=ni*(ni-1.0)*(ni-2.0)*nj;
    out[t]=(d>0.0)?(AQ3[i]*Q[j]+AS3[i]*S[j])/d:0.0;
}''', "k", options=("-std=c++11",))

_PI2_IJII_KERN = cp.RawKernel(r'''
extern "C" __global__ void k(const double*BP3,const double*BR3,
    const double*P,const double*R,const double*nn,
    const int*I,const int*J,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],j=J[t];
    double ni=nn[i],nj=nn[j];
    double d=ni*(ni-1.0)*(ni-2.0)*nj;
    out[t]=(d>0.0)?(BP3[i]*P[j]+BR3[i]*R[j])/d:0.0;
}''', "k", options=("-std=c++11",))

_PI2_IJIJ_KERN = cp.RawKernel(r'''
extern "C" __global__ void k(
    const double*P,const double*Q,const double*R,const double*S,
    const double*X01,const double*X10,const double*X11,const double*Bsgn,
    const double*nn,const int*I,const int*J,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=N)return;
    int i=I[t],j=J[t];
    double Pi=P[i],Qi=Q[i],Ri=R[i],Si=S[i],X01i=X01[i],X10i=X10[i],X11i=X11[i],Bi=Bsgn[i],ni=nn[i];
    double Pj=P[j],Qj=Q[j],Rj=R[j],Sj=S[j],X01j=X01[j],X10j=X10[j],X11j=X11[j],Bj=Bsgn[j],nj=nn[j];
    double PQi=Pi*Qi,PRi=Pi*Ri,PSi=Pi*Si,QRi=Qi*Ri,RSi=Ri*Si;
    double PQj=Pj*Qj,PSj=Pj*Sj,QRj=Qj*Rj,RSj=Rj*Sj;
    double Tcorri=Ri*(1.0+X01i-X10i),Ucorri=Si*(1.0+Ri);
    double num=0.0;
    num+=-0.25*RSi*Qj-0.25*X11i*PQj+0.25*RSi*PQj-0.25*Ri*PSj;
    num+=-0.25*PQi*X11j-0.25*RSi*X11j+0.25*PSi*X11j+0.25*QRi*X11j;
    num+=-0.25*PRi*Sj+0.25*Tcorri*Sj+0.25*X11i*PSj+0.25*QRi*PSj;
    num+=0.25*Ucorri*Rj-0.25*X01i*QRj+0.25*PSi*QRj;
    num+=-0.25*PSi*Rj-0.25*Pi*RSj+0.25*X01i*RSj+0.25*PQi*RSj;
    num+=0.25*X11i*Bj+0.25*Bi*X11j-1.0*X11i*X11j;
    double d=ni*(ni-1.0)*nj*(nj-1.0);
    out[t]=(d>0.0)?num/d:0.0;
}''', "k", options=("-std=c++11",))

_PI2_IIII_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double* __restrict__ g, double* __restrict__ out, const long long N){
    const long long idx=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    if(idx>=N)return;
    const double*row=g+9*idx;
    double n1=row[0],n2=row[1],n3=row[2],n4=row[3],n5=row[4],n6=row[5],n7=row[6],n8=row[7],n9=row[8];
    double ns=n1+n2+n3+n4+n5+n6+n7+n8+n9;
    if(ns<4.0){out[idx]=0.0;return;}
    double a1=n1+n2+n3+0.5*n4+0.5*n5+0.5*n6;
    double a2=n1+0.5*n2+n4+0.5*n5+n7+0.5*n8;
    double a3=0.5*n2+n3+0.5*n5+n6+0.5*n8+n9;
    double a4=0.5*n4+0.5*n5+0.5*n6+n7+n8+n9;
    double corr=(
        (13.0*n2*n4-16.0*n1*n2*n4-11.0*n2*n2*n4+16.0*n3*n4-28.0*n1*n3*n4-24.0*n2*n3*n4)+
        (-8.0*n3*n3*n4-11.0*n2*n4*n4-20.0*n3*n4*n4-6.0*n5+12.0*n1*n5-4.0*n1*n1*n5+17.0*n2*n5)+
        (-20.0*n1*n2*n5-11.0*n2*n2*n5+12.0*n3*n5-28.0*n1*n3*n5-20.0*n2*n3*n5-4.0*n3*n3*n5)+
        (17.0*n4*n5-20.0*n1*n4*n5-32.0*n2*n4*n5-40.0*n3*n4*n5-11.0*n4*n4*n5+11.0*n5*n5)+
        (-16.0*n1*n5*n5-17.0*n2*n5*n5-16.0*n3*n5*n5-17.0*n4*n5*n5-6.0*n5*n5*n5+16.0*n1*n6-8.0*n1*n1*n6)+
        (13.0*n2*n6-24.0*n1*n2*n6-11.0*n2*n2*n6-28.0*n1*n3*n6-16.0*n2*n3*n6+24.0*n4*n6)+
        (-36.0*n1*n4*n6-38.0*n2*n4*n6-36.0*n3*n4*n6-20.0*n4*n4*n6+17.0*n5*n6-40.0*n1*n5*n6)+
        (-32.0*n2*n5*n6-20.0*n3*n5*n6-42.0*n4*n5*n6-17.0*n5*n5*n6-20.0*n1*n6*n6-11.0*n2*n6*n6)+
        (-20.0*n4*n6*n6-11.0*n5*n6*n6+16.0*n2*n7-28.0*n1*n2*n7-20.0*n2*n2*n7+16.0*n3*n7)+
        (-48.0*n1*n3*n7-44.0*n2*n3*n7-16.0*n3*n3*n7-24.0*n2*n4*n7-44.0*n3*n4*n7)+
        (12.0*n5*n7-28.0*n1*n5*n7-40.0*n2*n5*n7-48.0*n3*n5*n7-20.0*n4*n5*n7-16.0*n5*n5*n7)+
        (16.0*n6*n7-48.0*n1*n6*n7-48.0*n2*n6*n7-44.0*n3*n6*n7-36.0*n4*n6*n7-40.0*n5*n6*n7)+
        (-20.0*n6*n6*n7-8.0*n2*n7*n7-16.0*n3*n7*n7-4.0*n5*n7*n7-8.0*n6*n7*n7+16.0*n1*n8-8.0*n1*n1*n8)+
        (24.0*n2*n8-36.0*n1*n2*n8-20.0*n2*n2*n8+16.0*n3*n8-48.0*n1*n3*n8-36.0*n2*n3*n8-8.0*n3*n3*n8)+
        (13.0*n4*n8-24.0*n1*n4*n8-38.0*n2*n4*n8-48.0*n3*n4*n8-11.0*n4*n4*n8+17.0*n5*n8-40.0*n1*n5*n8)+
        (-42.0*n2*n5*n8-40.0*n3*n5*n8-32.0*n4*n5*n8-17.0*n5*n5*n8+13.0*n6*n8-48.0*n1*n6*n8)+
        (-38.0*n2*n6*n8-24.0*n3*n6*n8-38.0*n4*n6*n8-32.0*n5*n6*n8-11.0*n6*n6*n8-28.0*n1*n7*n8)+
        (-36.0*n2*n7*n8-44.0*n3*n7*n8-16.0*n4*n7*n8-20.0*n5*n7*n8-24.0*n6*n7*n8-20.0*n1*n8*n8)+
        (-20.0*n2*n8*n8-20.0*n3*n8*n8-11.0*n4*n8*n8-11.0*n5*n8*n8-11.0*n6*n8*n8+16.0*n1*n9-16.0*n1*n1*n9)+
        (16.0*n2*n9-44.0*n1*n2*n9-20.0*n2*n2*n9-48.0*n1*n3*n9-28.0*n2*n3*n9+16.0*n4*n9)+
        (-44.0*n1*n4*n9-48.0*n2*n4*n9-48.0*n3*n4*n9-20.0*n4*n4*n9+12.0*n5*n9-48.0*n1*n5*n9)+
        (-40.0*n2*n5*n9-28.0*n3*n5*n9-40.0*n4*n5*n9-16.0*n5*n5*n9-44.0*n1*n6*n9-24.0*n2*n6*n9)+
        (-36.0*n4*n6*n9-20.0*n5*n6*n9-48.0*n1*n7*n9-48.0*n2*n7*n9-48.0*n3*n7*n9-28.0*n4*n7*n9)+
        (-28.0*n5*n7*n9-28.0*n6*n7*n9-44.0*n1*n8*n9-36.0*n2*n8*n9-28.0*n3*n8*n9-24.0*n4*n8*n9)+
        (-20.0*n5*n8*n9-16.0*n6*n8*n9-16.0*n1*n9*n9-8.0*n2*n9*n9-8.0*n4*n9*n9-4.0*n5*n9*n9)
    )*(1.0/16.0);
    out[idx]=(a1*a2*a3*a4+corr)/(ns*(ns-1.0)*(ns-2.0)*(ns-3.0));
}''', "k", options=("-std=c++11",))


# ---------------------------------------------------------------------------
# _PopDataGeno -- precomputed per-population arrays
# ---------------------------------------------------------------------------

class _PopDataGeno:
    """Precomputed per-population arrays for genotype-based LD computation.

    Stores 9 genotype configuration counts and derived frequency quantities.
    """
    __slots__ = ('g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9',
                 'n', 'D_geno', 'pA', 'qA', 'pB', 'qB', 'fdA', 'fdB',
                 'B_L', 'B_R', 'C00', 'C01', 'C10', 'C11',
                 'AQ3', 'AS3', 'BP3', 'BR3',
                 'X01', 'X10', 'X11', 'Bsgn')

    def __init__(self, counts, n_valid):
        # Our counting: combo = geno_i*3+geno_j gives columns:
        # col0=n00, col1=n01, col2=n02, col3=n10, col4=n11, col5=n12, col6=n20, col7=n21, col8=n22
        # Moments convention: g1=n22, g2=n21, g3=n20, g4=n12, g5=n11, g6=n10, g7=n02, g8=n01, g9=n00
        self.g1 = counts[:, 8].astype(cp.float64)  # n22
        self.g2 = counts[:, 7].astype(cp.float64)  # n21
        self.g3 = counts[:, 6].astype(cp.float64)  # n20
        self.g4 = counts[:, 5].astype(cp.float64)  # n12
        self.g5 = counts[:, 4].astype(cp.float64)  # n11
        self.g6 = counts[:, 3].astype(cp.float64)  # n10
        self.g7 = counts[:, 2].astype(cp.float64)  # n02
        self.g8 = counts[:, 1].astype(cp.float64)  # n01
        self.g9 = counts[:, 0].astype(cp.float64)  # n00
        if n_valid is not None:
            self.n = n_valid.astype(cp.float64)
        else:
            self.n = (self.g1 + self.g2 + self.g3 + self.g4 + self.g5
                      + self.g6 + self.g7 + self.g8 + self.g9)
        # D from genotype frequencies (used in between-pop DD, Dz all-diff)
        self.D_geno = (
            -(self.g2 / 2 + self.g3 + self.g5 / 4 + self.g6 / 2)
            * (self.g4 / 2 + self.g5 / 4 + self.g7 + self.g8 / 2)
            + (self.g1 + self.g2 / 2 + self.g4 / 2 + self.g5 / 4)
            * (self.g5 / 4 + self.g6 / 2 + self.g8 / 2 + self.g9)
        )
        # Allele frequency terms for between-pop formulas
        self.pA = self.g1 + self.g2 + self.g3 + self.g4 / 2 + self.g5 / 2 + self.g6 / 2
        self.qA = self.g4 / 2 + self.g5 / 2 + self.g6 / 2 + self.g7 + self.g8 + self.g9
        self.pB = self.g1 + self.g2 / 2 + self.g4 + self.g5 / 2 + self.g7 + self.g8 / 2
        self.qB = self.g2 / 2 + self.g3 + self.g5 / 2 + self.g6 + self.g8 / 2 + self.g9
        self.fdA = -self.g1 + self.g3 - self.g4 + self.g6 - self.g7 + self.g9
        self.fdB = -self.g1 - self.g2 - self.g3 + self.g7 + self.g8 + self.g9
        # pi2 kernel precomputations
        H_L = self.g4 + self.g5 + self.g6  # het count at locus A
        H_R = self.g2 + self.g5 + self.g8  # het count at locus B
        nn1 = 4 * self.n * (self.n - 1)
        self.B_L = (4*self.pA*self.qA - H_L) / cp.maximum(nn1, 1)
        self.B_R = (4*self.pB*self.qB - H_R) / cp.maximum(nn1, 1)
        # Same-individual joint allele matrix X (2x2 per pair)
        x_AB = self.g1 + self.g2/2 + self.g4/2 + self.g5/4
        x_Ab = self.g2/2 + self.g3 + self.g5/4 + self.g6/2
        x_aB = self.g4/2 + self.g5/4 + self.g7 + self.g8/2
        x_ab = self.g5/4 + self.g6/2 + self.g8/2 + self.g9
        # C_i = ([P,R]^T [Q,S] - X) / (4*n*(n-1))
        self.C00 = (self.pA*self.pB - x_AB) / cp.maximum(nn1, 1)
        self.C01 = (self.pA*self.qB - x_Ab) / cp.maximum(nn1, 1)
        self.C10 = (self.qA*self.pB - x_aB) / cp.maximum(nn1, 1)
        self.C11 = (self.qA*self.qB - x_ab) / cp.maximum(nn1, 1)
        # ijij features
        self.X01 = x_Ab
        self.X10 = x_aB
        self.X11 = x_ab
        self.Bsgn = x_AB - x_Ab - x_aB + x_ab
        # Triple coefficients: pi2(i,i;i,j) uses AQ3,AS3; pi2(i,j;i,i) uses BP3,BR3
        P, R, Q, S = self.pA, self.qA, self.pB, self.qB
        L0 = self.g5 + 2*self.g6 + 2*self.g8 + 4*self.g9
        L1 = self.g5 + 2*self.g6
        M1 = self.g5 + 2*self.g8
        negR3 = 3*self.g4 + 3*self.g5 + 3*self.g6 + 4*self.g7 + 4*self.g8 + 4*self.g9
        negR5 = 5*self.g4 + 5*self.g5 + 5*self.g6 + 8*self.g7 + 8*self.g8 + 8*self.g9
        beta5A = -2*self.g2 - 4*self.g3 + 4*self.g4 + self.g5 - 2*self.g6 + 8*self.g7 + 4*self.g8 + 4
        beta6A = -self.g5 - 2*self.g6 - 4*self.g7 - 4*self.g8 - 4*self.g9
        self.AQ3 = -(S*negR3)/8 + P*S*R/2 + (R*L0)/8 + L1/8 - (P*L0)/8
        self.AS3 = -(Q*negR5)/8 + P*Q*R/2 + (R*beta5A)/8 + beta6A/8 + (P*L0)/8
        negL3 = 3*self.g2 + 4*self.g3 + 3*self.g5 + 4*self.g6 + 3*self.g8 + 4*self.g9
        negL5 = 5*self.g2 + 8*self.g3 + 5*self.g5 + 8*self.g6 + 5*self.g8 + 8*self.g9
        beta5B = 4*self.g2 + 8*self.g3 - 2*self.g4 + self.g5 + 4*self.g6 - 4*self.g7 - 2*self.g8 + 4
        beta6B = -4*self.g3 - self.g5 - 4*self.g6 - 2*self.g8 - 4*self.g9
        self.BP3 = -(R*negL3)/8 + Q*S*R/2 + (S*L0)/8 + M1/8 - (Q*L0)/8
        self.BR3 = -(P*negL5)/8 + P*Q*S/2 + (S*beta5B)/8 + beta6B/8 + (Q*L0)/8


# ---------------------------------------------------------------------------
# _GenoPopFlat -- flattened per-pop arrays for fused kernel indexing
# ---------------------------------------------------------------------------

class _GenoPopFlat:
    """Flattened per-pop arrays for fused genotype kernel indexing.

    Mirrors _HapPopFlat: concatenates all per-pop feature arrays into
    contiguous buffers so that flat_idx = pop * N_pairs + pair gives
    direct kernel access. Used by the pi2 dispatch function.
    """
    __slots__ = ('p', 'r', 'q', 's', 'BL', 'BR',
                 'C00', 'C01', 'C10', 'C11',
                 'P', 'Q', 'R', 'S', 'n',
                 'X01', 'X10', 'X11', 'Bsgn',
                 'AQ3', 'AS3', 'BP3', 'BR3',
                 'g_moments')

    def __init__(self, pops):
        P = len(pops)

        def flat(arrs):
            return cp.ascontiguousarray(cp.concatenate(arrs))

        inv_n = [1.0 / p.n for p in pops]
        # Normalized frequencies for between-pop factored kernels
        self.p = flat([pops[i].pA * inv_n[i] for i in range(P)])
        self.r = flat([pops[i].qA * inv_n[i] for i in range(P)])
        self.q = flat([pops[i].pB * inv_n[i] for i in range(P)])
        self.s = flat([pops[i].qB * inv_n[i] for i in range(P)])
        # Bias-corrected within-pop heterozygosity products
        self.BL = flat([p.B_L for p in pops])
        self.BR = flat([p.B_R for p in pops])
        # Covariance matrix entries for shared-pop kernel
        self.C00 = flat([p.C00 for p in pops])
        self.C01 = flat([p.C01 for p in pops])
        self.C10 = flat([p.C10 for p in pops])
        self.C11 = flat([p.C11 for p in pops])
        # Raw allele counts for ijij and triple kernels
        self.P = flat([p.pA for p in pops])
        self.Q = flat([p.pB for p in pops])
        self.R = flat([p.qA for p in pops])
        self.S = flat([p.qB for p in pops])
        self.n = flat([p.n for p in pops])
        # ijij kernel features
        self.X01 = flat([p.X01 for p in pops])
        self.X10 = flat([p.X10 for p in pops])
        self.X11 = flat([p.X11 for p in pops])
        self.Bsgn = flat([p.Bsgn for p in pops])
        # Triple kernel coefficients
        self.AQ3 = flat([p.AQ3 for p in pops])
        self.AS3 = flat([p.AS3 for p in pops])
        self.BP3 = flat([p.BP3 for p in pops])
        self.BR3 = flat([p.BR3 for p in pops])
        # Genotype counts in moments order for single-pop fused kernel
        self.g_moments = flat([
            cp.stack([p.g1, p.g2, p.g3, p.g4, p.g5, p.g6, p.g7, p.g8, p.g9],
                     axis=-1)
            for p in pops
        ])  # (P*N, 9)


# ---------------------------------------------------------------------------
# Batch dispatch functions
# ---------------------------------------------------------------------------

def compute_all_dd_geno(pops, dd_calls):
    """Compute all DD values for genotype data.

    DD has only ~10 calls even for 4 populations, so we dispatch directly
    to the ld_statistics_genotype polynomial formulas rather than using
    fused CUDA kernels.
    """
    from . import ld_statistics_genotype as ldg

    results = []
    for i, j in dd_calls:
        if i == j:
            results.append(ldg.dd_geno_single(pops[i]))
        else:
            results.append(ldg.dd_geno_between(pops[i], pops[j]))
    return results


def compute_all_dz_geno(pops, dz_calls):
    """Fused Dz computation for all calls using genotype kernels."""
    P = len(pops)
    N = pops[0].n.shape[0]

    # Build g in moments order and precompute features via Dz precompute kernel
    g_moments = cp.ascontiguousarray(cp.stack([
        cp.stack([p.g1, p.g2, p.g3, p.g4, p.g5, p.g6, p.g7, p.g8, p.g9],
                 axis=-1)
        for p in pops
    ]).reshape(P * N, 9))

    ns_f = cp.empty(P * N, dtype=cp.float64)
    D_f = cp.empty(P * N, dtype=cp.float64)
    A_f = cp.empty(P * N, dtype=cp.float64)
    Bv_f = cp.empty(P * N, dtype=cp.float64)
    Wjj_f = cp.empty(P * N, dtype=cp.float64)
    Ucoef_f = cp.empty(P * N, dtype=cp.float64)
    Vcoef_f = cp.empty(P * N, dtype=cp.float64)
    _launch(_DZ_PRECOMP_KERN,
            (g_moments, ns_f, D_f, A_f, Bv_f, Wjj_f, Ucoef_f, Vcoef_f,
             P * N),
            P * N)

    # Group calls by case type
    groups = {'same': [], 'p1p2': [], 'p1p3': [], 'p2p3': [], 'diff': []}
    call_order = {}
    for idx, (p1, p2, p3) in enumerate(dz_calls):
        if p1 == p2 == p3:
            key = 'same'
        elif p1 == p2:
            key = 'p1p2'
        elif p1 == p3:
            key = 'p1p3'
        elif p2 == p3:
            key = 'p2p3'
        else:
            key = 'diff'
        call_order[idx] = (key, len(groups[key]))
        groups[key].append((p1, p2, p3))

    pair_range = cp.arange(N, dtype=cp.int32)

    def expand(pop_arr):
        return (pop_arr[:, None] * N + pair_range[None, :]).ravel()

    group_results = {}

    for case in groups:
        calls = groups[case]
        if not calls:
            continue
        n_c = len(calls)

        if case == 'diff':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
            fK = expand(cp.array([c[2] for c in calls], dtype=cp.int32))
            M = n_c * N
            out = cp.empty(M, dtype=cp.float64)
            _launch(_DZ_DISTINCT_KERN,
                    (ns_f, D_f, A_f, Bv_f, fI, fJ, fK, out, M), M)

        elif case == 'p1p2':  # Dz(i,i,j): repeated=p1, singleton=p3
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[2] for c in calls], dtype=cp.int32))
            M = n_c * N
            out = cp.empty(M, dtype=cp.float64)
            _launch(_DZ_IIJ_KERN,
                    (ns_f, Ucoef_f, A_f, fI, fJ, out, M), M)

        elif case == 'p1p3':  # Dz(i,j,i): repeated=p1, singleton=p2
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
            M = n_c * N
            out = cp.empty(M, dtype=cp.float64)
            _launch(_DZ_IJI_KERN,
                    (ns_f, Vcoef_f, Bv_f, fI, fJ, out, M), M)

        elif case == 'p2p3':  # Dz(i,j,j): single=p1, repeated=p2
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
            M = n_c * N
            out = cp.empty(M, dtype=cp.float64)
            _launch(_DZ_IJJ_KERN,
                    (ns_f, D_f, Wjj_f, fI, fJ, out, M), M)

        elif case == 'same':
            M = P * N  # compute for ALL pops, extract needed ones later
            out_all = cp.empty(M, dtype=cp.float64)
            _launch(_DZ_III_KERN, (g_moments, out_all, M), M)
            # Extract per-pop results
            out_per_pop = out_all.reshape(P, N)
            group_results[case] = [out_per_pop[c[0]] for c in calls]
            continue

        group_results[case] = [out[c_idx * N:(c_idx + 1) * N]
                               for c_idx in range(n_c)]

    output = [None] * len(dz_calls)
    for idx in range(len(dz_calls)):
        grp, pos = call_order[idx]
        output[idx] = group_results[grp][pos]
    return output


def compute_all_pi2_geno(pops, pi2_calls):
    """Fused pi2 computation for all calls using genotype kernels.

    All 9 pi2 case types use fused RawKernels that compute one output
    per thread with all arithmetic in registers.
    """
    P = len(pops)
    N = pops[0].n.shape[0]
    F = _GenoPopFlat(pops)

    # Group calls by case type
    groups = {k: [] for k in
              ['same', 'triple_iiij', 'triple_ijii', 'iikk', 'iikl',
               'ijkk', 'ijij', 'shared', 'alldiff']}
    call_order = {}
    for idx, (i, j, k, l) in enumerate(pi2_calls):
        cnt = {}
        for pp in (i, j, k, l):
            cnt[pp] = cnt.get(pp, 0) + 1
        nu = len(cnt)
        mc = max(cnt.values())
        if nu == 1:
            key = 'same'
        elif mc == 3:
            tp = [pp for pp, c in cnt.items() if c == 3][0]
            key = 'triple_iiij' if (i == tp and j == tp) else 'triple_ijii'
        elif i == j and k == l:
            key = 'iikk'
        elif i == j:
            key = 'iikl'
        elif k == l:
            key = 'ijkk'
        elif (i == k and j == l) or (i == l and j == k):
            key = 'ijij'
        elif nu == 3:
            key = 'shared'
        else:
            key = 'alldiff'
        call_order[idx] = (key, len(groups[key]))
        groups[key].append((i, j, k, l))

    pair_range = cp.arange(N, dtype=cp.int32)

    def expand(pop_arr):
        """Expand pop indices to flat pair indices: pop*N + pair."""
        return (pop_arr[:, None] * N + pair_range[None, :]).ravel()

    def pop_arr(calls, pos):
        return cp.array([c[pos] for c in calls], dtype=cp.int32)

    def triple_pop(calls):
        """Extract (triple_pop, single_pop) for each call."""
        tp_list, sp_list = [], []
        for c in calls:
            cnt = {}
            for pp in c:
                cnt[pp] = cnt.get(pp, 0) + 1
            tp_list.append([pp for pp, cc in cnt.items() if cc == 3][0])
            sp_list.append([pp for pp, cc in cnt.items() if cc == 1][0])
        return (cp.array(tp_list, dtype=cp.int32),
                cp.array(sp_list, dtype=cp.int32))

    group_results = {}

    # Launch each case type as a single fused kernel
    for case in groups:
        calls = groups[case]
        if not calls:
            continue
        n_c = len(calls)
        M = n_c * N
        out = cp.empty(M, dtype=cp.float64)

        if case == 'alldiff':
            fI, fJ = expand(pop_arr(calls, 0)), expand(pop_arr(calls, 1))
            fK, fL = expand(pop_arr(calls, 2)), expand(pop_arr(calls, 3))
            _launch(_PI2_ALLDIFF_KERN,
                    (F.p, F.r, F.q, F.s, fI, fJ, fK, fL, out, M), M)

        elif case == 'iikk':
            fI, fK = expand(pop_arr(calls, 0)), expand(pop_arr(calls, 2))
            _launch(_PI2_IIKK_KERN,
                    (F.BL, F.BR, fI, fK, out, M), M)

        elif case == 'iikl':
            fI = expand(pop_arr(calls, 0))
            fK, fL = expand(pop_arr(calls, 2)), expand(pop_arr(calls, 3))
            _launch(_PI2_IIKL_KERN,
                    (F.BL, F.q, F.s, fI, fK, fL, out, M), M)

        elif case == 'ijkk':
            fI, fJ = expand(pop_arr(calls, 0)), expand(pop_arr(calls, 1))
            fK = expand(pop_arr(calls, 2))
            _launch(_PI2_IJKK_KERN,
                    (F.p, F.r, F.BR, fI, fJ, fK, out, M), M)

        elif case == 'shared':
            si_l, ai_l, bi_l = [], [], []
            for i, j, k, l in calls:
                if i == k:
                    si, ai, bi = i, j, l
                elif i == l:
                    si, ai, bi = i, j, k
                elif j == k:
                    si, ai, bi = j, i, l
                elif j == l:
                    si, ai, bi = j, i, k
                else:
                    si, ai, bi = 0, 0, 0
                si_l.append(si)
                ai_l.append(ai)
                bi_l.append(bi)
            fI = expand(cp.array(si_l, dtype=cp.int32))
            fJ = expand(cp.array(ai_l, dtype=cp.int32))
            fK = expand(cp.array(bi_l, dtype=cp.int32))
            _launch(_PI2_SHARED_KERN,
                    (F.p, F.r, F.q, F.s,
                     F.C00, F.C01, F.C10, F.C11,
                     fI, fJ, fK, out, M), M)

        elif case == 'triple_iiij':
            tp_a, sp_a = triple_pop(calls)
            fI, fJ = expand(tp_a), expand(sp_a)
            _launch(_PI2_IIIJ_KERN,
                    (F.AQ3, F.AS3, F.Q, F.S, F.n,
                     fI, fJ, out, M), M)

        elif case == 'triple_ijii':
            tp_a, sp_a = triple_pop(calls)
            fI, fJ = expand(tp_a), expand(sp_a)
            _launch(_PI2_IJII_KERN,
                    (F.BP3, F.BR3, F.P, F.R, F.n,
                     fI, fJ, out, M), M)

        elif case == 'ijij':
            fI, fJ = expand(pop_arr(calls, 0)), expand(pop_arr(calls, 1))
            _launch(_PI2_IJIJ_KERN,
                    (F.P, F.Q, F.R, F.S,
                     F.X01, F.X10, F.X11, F.Bsgn, F.n,
                     fI, fJ, out, M), M)

        elif case == 'same':
            # Fused single-pop kernel operates on (M, 9) genotype counts
            pop_indices = [c[0] for c in calls]
            g_rows = cp.stack([F.g_moments[p * N:(p + 1) * N]
                               for p in pop_indices])
            g_flat = cp.ascontiguousarray(g_rows.reshape(M, 9))
            _PI2_IIII_KERN(((M + 255) // 256,), (256,), (g_flat, out, M))

        group_results[case] = [out[c * N:(c + 1) * N] for c in range(n_c)]

    output = [None] * len(pi2_calls)
    for idx in range(len(pi2_calls)):
        grp, pos = call_order[idx]
        output[idx] = group_results[grp][pos]
    return output


def compute_multi_pop_statistics_batch_geno(counts_per_pop, n_valid_per_pop,
                                            _, stat_specs):
    """Compute all LD statistics for N populations using genotype counts.

    Drop-in replacement for _compute_multi_pop_statistics_batch but uses
    9-way genotype counts and genotype-specific formulas.
    """
    n_pairs = counts_per_pop[0].shape[0]
    n_stats = len(stat_specs)

    pops = [_PopDataGeno(counts_per_pop[p], n_valid_per_pop[p])
            for p in range(len(counts_per_pop))]

    # Collect unique calls
    unique_calls = set()
    for _, calls in stat_specs:
        for _, stat_type, pop_indices in calls:
            unique_calls.add((stat_type, pop_indices))

    raw_cache = {}

    # DD -- few calls, compute individually via ldg formula functions
    dd_calls = [pi for st, pi in unique_calls if st == 'dd']
    if dd_calls:
        dd_results = compute_all_dd_geno(pops, dd_calls)
        for pi, val in zip(dd_calls, dd_results):
            raw_cache[('dd', pi)] = val

    # Dz -- batch via fused kernels
    dz_calls = [pi for st, pi in unique_calls if st == 'dz']
    if dz_calls:
        dz_results = compute_all_dz_geno(pops, dz_calls)
        for pi, val in zip(dz_calls, dz_results):
            raw_cache[('dz', pi)] = val

    # pi2 -- batch via fused kernels
    pi2_calls = [pi for st, pi in unique_calls if st == 'pi2']
    if pi2_calls:
        pi2_results = compute_all_pi2_geno(pops, pi2_calls)
        for pi, val in zip(pi2_calls, pi2_results):
            raw_cache[('pi2', pi)] = val

    # Assemble weighted sums
    result = cp.zeros((n_pairs, n_stats), dtype=cp.float64)
    for stat_idx, (_, calls) in enumerate(stat_specs):
        if len(calls) == 1:
            w, st, pi = calls[0]
            if w == 1.0:
                result[:, stat_idx] = raw_cache[(st, pi)]
            else:
                result[:, stat_idx] = w * raw_cache[(st, pi)]
        else:
            val = sum(w * raw_cache[(st, pi)] for w, st, pi in calls)
            result[:, stat_idx] = val

    return result
