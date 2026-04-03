"""
Fused CUDA kernels for haplotype-based LD statistics (DD, Dz, pi2).

All multi-population cases use RawKernels that compute one output per GPU
thread with all arithmetic in registers. Haplotype counts use 4-way
layout: (n11, n10, n01, n00) per locus pair per population.
"""
import cupy as cp


def _launch(kernel, args, M):
    """Launch a fused kernel with M output elements."""
    kernel(((int(M) + 255) // 256,), (256,), args)


# ---------------------------------------------------------------------------
# DD kernels
# ---------------------------------------------------------------------------

_DD_SINGLE_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1, const double*c2, const double*c3, const double*c4,
       const double*nn, const int*I, double*out, const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t];
    double a=c1[i],b=c2[i],c=c3[i],d=c4[i],n=nn[i];
    double num=a*(a-1.)*d*(d-1.)+b*(b-1.)*c*(c-1.)-2.*a*b*c*d;
    double den=n*(n-1.)*(n-2.)*(n-3.);
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_DD_BETWEEN_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*D, const double*nn,
       const int*I, const int*J, double*out, const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],j=J[t];
    double den=nn[i]*(nn[i]-1.)*nn[j]*(nn[j]-1.);
    out[t]=(den>0.)?D[i]*D[j]/den:0.;
}''', "k", options=("-std=c++11",))

# ---------------------------------------------------------------------------
# Dz kernels
# ---------------------------------------------------------------------------

_DZ_III_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1,const double*c2,const double*c3,const double*c4,
       const double*nn, const int*I, double*out, const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t];
    double a=c1[i],b=c2[i],c=c3[i],d=c4[i],n=nn[i];
    double diff=a*d-b*c;
    double sqa=c+d-a-b, sqb=b+d-a-c, sqc=b+c-a-d;
    double num=diff*sqa*sqb+diff*sqc+2.*(b*c+a*d);
    double den=n*(n-1.)*(n-2.)*(n-3.);
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_DZ_IIJ_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1,const double*c2,const double*c3,const double*c4,
       const double*nn,const double*pA,const double*qA,
       const int*I,const int*J,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],j=J[t];
    double D=c2[i]*c3[i]-c1[i]*c4[i];
    double num=(-pA[i]+qA[i])*(-D)*(-c1[j]+c2[j]-c3[j]+c4[j]);
    double den=nn[j]*nn[i]*(nn[i]-1.)*(nn[i]-2.);
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_DZ_IJI_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1,const double*c2,const double*c3,const double*c4,
       const double*nn,const double*pA,const double*qA,
       const int*I,const int*J,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],j=J[t];
    double D=c2[i]*c3[i]-c1[i]*c4[i];
    double num=(-c1[i]+c2[i]-c3[i]+c4[i])*(-D)*(-pA[j]+qA[j]);
    double den=nn[j]*nn[i]*(nn[i]-1.)*(nn[i]-2.);
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_DZ_IJJ_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1,const double*c2,const double*c3,const double*c4,
       const double*nn,const double*pA,const double*qA,
       const int*I,const int*J,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],j=J[t];
    double Di=c2[i]*c3[i]-c1[i]*c4[i];
    double num=(-Di)*(-c1[j]+c2[j]+c3[j]-c4[j])
              +(-Di)*(-c1[j]+c2[j]-c3[j]+c4[j])*(-pA[j]+qA[j]);
    double den=nn[i]*(nn[i]-1.)*nn[j]*(nn[j]-1.);
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_DZ_DIFF_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*D,const double*nn,
       const double*pA,const double*qA,
       const double*c1,const double*c2,const double*c3,const double*c4,
       const int*I,const int*J,const int*K,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],j=J[t],k=K[t];
    double num=-(D[i]*(pA[j]-qA[j])*(c1[k]-c2[k]+c3[k]-c4[k]));
    double den=nn[i]*(nn[i]-1.)*nn[j]*nn[k];
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

# ---------------------------------------------------------------------------
# pi2 kernels
# ---------------------------------------------------------------------------

# Factored cases use per-pop frequency products
_PI2_ALLDIFF_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*pA,const double*qA,const double*pB,const double*qB,
       const double*nn,
       const int*I,const int*J,const int*K,const int*L,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],j=J[t],k=K[t],l=L[t];
    double HA=(pA[i]*qA[j]+qA[i]*pA[j]);
    double HB=(pB[k]*qB[l]+qB[k]*pB[l]);
    double den=nn[i]*nn[j]*nn[k]*nn[l];
    out[t]=(den>0.)?HA*HB/(4.*den):0.;
}''', "k", options=("-std=c++11",))

_PI2_IIKK_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*pA,const double*qA,const double*pB,const double*qB,
       const double*nn,
       const int*I,const int*K,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],k=K[t];
    double fA=pA[i]*qA[i], fB=pB[k]*qB[k];
    double den=nn[i]*(nn[i]-1.)*nn[k]*(nn[k]-1.);
    out[t]=(den>0.)?fA*fB/den:0.;
}''', "k", options=("-std=c++11",))

_PI2_IIKL_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*pA,const double*qA,const double*pB,const double*qB,
       const double*nn,
       const int*I,const int*K,const int*L,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],k=K[t],l=L[t];
    double fA=pA[i]*qA[i];
    double HB=(pB[k]*qB[l]+qB[k]*pB[l]);
    double den=nn[i]*(nn[i]-1.)*nn[k]*nn[l];
    out[t]=(den>0.)?fA*HB/(2.*den):0.;
}''', "k", options=("-std=c++11",))

_PI2_IJKK_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*pA,const double*qA,const double*pB,const double*qB,
       const double*nn,
       const int*I,const int*J,const int*K,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],j=J[t],k=K[t];
    double HA=(pA[i]*qA[j]+qA[i]*pA[j]);
    double fB=pB[k]*qB[k];
    double den=nn[k]*(nn[k]-1.)*nn[i]*nn[j];
    out[t]=(den>0.)?fB*HA/(2.*den):0.;
}''', "k", options=("-std=c++11",))

_PI2_SHARED_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1,const double*c2,const double*c3,const double*c4,
       const double*nn,const double*pA,const double*qA,const double*pB,const double*qB,
       const int*I,const int*J,const int*K,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int s=I[t],a=J[t],b=K[t];
    double cs1=c1[s],cs2=c2[s],cs3=c3[s],cs4=c4[s],ns=nn[s];
    double ca1=c1[a],ca2=c2[a],cb1=c1[b],cb2=c2[b],cb3=c3[b],cb4=c4[b];
    double na=nn[a],nb=nn[b];
    double num=(
        cs4*cs4*(pA[a])*(pB[b])
        +cs2*cs2*(qA[a])*(pB[b])
        +(-1.+cs1+cs3)*(cs3*(pA[a])+cs1*(qA[a]))*(qB[b])
        +cs4*(cs1*(qA[a])*(pB[b])
              +ca1*((-1.+cs3)*cb1+cs3*cb2-cb3+cs3*cb3+cs3*cb4+cs1*(qB[b]))
              +ca2*((-1.+cs3)*cb1+cs3*cb2-cb3+cs3*cb3+cs3*cb4+cs1*(qB[b])))
        +cs2*(cs4*((pA[a])+(qA[a]))*(pB[b])
              +cs3*(ca1*(pB[b])+ca2*(pB[b])+(qA[a])*(qB[b]))
              +(qA[a])*((-1.+cs1)*cb1-cb3+cs1*(cb2+cb3+cb4)))
    )/4.;
    double den=ns*(ns-1.)*na*nb;
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_PI2_IJIJ_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1,const double*c2,const double*c3,const double*c4,
       const double*nn,const double*pA,const double*qA,const double*pB,const double*qB,
       const int*I,const int*J,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],j=J[t];
    double a1=c1[i],a2=c2[i],a3=c3[i],a4=c4[i];
    double b1=c1[j],b2=c2[j],b3=c3[j],b4=c4[j];
    double qBa=a2+a4,qAa=a3+a4,pAb=b1+b2,pBb=b1+b3;
    double pBa=a1+a3,qAb=b3+b4,qBb=b2+b4,pAa=a1+a2;
    double num=
        (qBa*qAa*pAb*pBb)/4.
        +(pBa*qAa*pAb*qBb)/4.
        +(pAa*qBa*pBb*qAb)/4.
        +(pAa*pBa*qBb*qAb)/4.
        +(-(a2*a3*b1)+a4*b1-a2*a4*b1-a3*a4*b1-a4*a4*b1-a4*b1*b1
          +a3*b2-a1*a3*b2-a3*a3*b2-a1*a4*b2-a3*a4*b2-a3*b1*b2-a4*b1*b2-a3*b2*b2
          +a2*b3-a1*a2*b3-a2*a2*b3-a1*a4*b3-a2*a4*b3-a2*b1*b3-a4*b1*b3-a1*b2*b3-a4*b2*b3-a2*b3*b3
          +a1*b4-a1*a1*b4-a1*a2*b4-a1*a3*b4-a2*a3*b4-a2*b1*b4-a3*b1*b4-a1*b2*b4-a3*b2*b4-a1*b3*b4-a2*b3*b4-a1*b4*b4
        )/4.;
    double den=nn[i]*(nn[i]-1.)*nn[j]*(nn[j]-1.);
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_PI2_TRIPLE_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1,const double*c2,const double*c3,const double*c4,
       const double*nn,const double*pA,const double*qA,const double*pB,const double*qB,
       const int*I,const int*J,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int tp=I[t],sp=J[t];
    double t1=c1[tp],t2=c2[tp],t3=c3[tp],t4=c4[tp],nt=nn[tp];
    double tpA=t1+t2,tqA=t3+t4;
    double spB=pB[sp],sqB=qB[sp];
    double num=(
        -(tpA*t4*spB)
        -(t2*tqA*spB)
        +(tpA*(t2+t4)*tqA*spB)
        +(tpA*tqA*(-2.*c2[sp]-2.*c4[sp]))
        +(tpA*t4*sqB)
        +(t2*tqA*sqB)
        +(tpA*(t1+t3)*tqA*sqB)
    )/2.;
    double den=nn[sp]*nt*(nt-1.)*(nt-2.);
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_PI2_IIII_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1,const double*c2,const double*c3,const double*c4,
       const double*nn,const int*I,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t];
    double a=c1[i],b=c2[i],c=c3[i],d=c4[i],n=nn[i];
    double pA=a+b,pB=a+c,qB=b+d,qA=c+d;
    double num=pA*pB*qB*qA
        -a*d*(-1.+a+3.*b+3.*c+d)
        -b*c*(-1.+3.*a+b+c+3.*d);
    double den=n*(n-1.)*(n-2.)*(n-3.);
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))


# ---------------------------------------------------------------------------
# _HapPopFlat -- flattened per-pop arrays for fused kernel indexing
# ---------------------------------------------------------------------------


class _HapPopFlat:
    """Flattened per-pop arrays for fused haplotype kernel indexing."""
    __slots__ = ('c1', 'c2', 'c3', 'c4', 'n', 'D', 'pA', 'qA', 'pB', 'qB')

    def __init__(self, pops):
        P = len(pops)
        self.c1 = cp.concatenate([p.c1 for p in pops])
        self.c2 = cp.concatenate([p.c2 for p in pops])
        self.c3 = cp.concatenate([p.c3 for p in pops])
        self.c4 = cp.concatenate([p.c4 for p in pops])
        self.n = cp.concatenate([p.n for p in pops])
        self.D = cp.concatenate([p.D for p in pops])
        self.pA = cp.concatenate([p.pA for p in pops])
        self.qA = cp.concatenate([p.qA for p in pops])
        self.pB = cp.concatenate([p.pB for p in pops])
        self.qB = cp.concatenate([p.qB for p in pops])


# ---------------------------------------------------------------------------
# Batch dispatch functions
# ---------------------------------------------------------------------------


def compute_all_dd_hap(pops, dd_calls):
    """Fused DD computation for all calls."""
    P = len(pops)
    N = pops[0].n.shape[0]
    F = _HapPopFlat(pops)
    pair_range = cp.arange(N, dtype=cp.int32)

    def expand(pop_arr):
        return (pop_arr[:, None] * N + pair_range[None, :]).ravel()

    # Separate single-pop and between-pop
    single_calls = [(idx, c) for idx, c in enumerate(dd_calls) if c[0] == c[1]]
    between_calls = [(idx, c) for idx, c in enumerate(dd_calls) if c[0] != c[1]]

    results = [None] * len(dd_calls)

    if single_calls:
        idxs = [i for i, _ in single_calls]
        calls = [c for _, c in single_calls]
        fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
        M = len(calls) * N
        out = cp.empty(M, dtype=cp.float64)
        _launch(_DD_SINGLE_KERN, (F.c1, F.c2, F.c3, F.c4, F.n, fI, out, M), M)
        for ci, orig_idx in enumerate(idxs):
            results[orig_idx] = out[ci * N:(ci + 1) * N]

    if between_calls:
        idxs = [i for i, _ in between_calls]
        calls = [c for _, c in between_calls]
        fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
        fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
        M = len(calls) * N
        out = cp.empty(M, dtype=cp.float64)
        _launch(_DD_BETWEEN_KERN, (F.D, F.n, fI, fJ, out, M), M)
        for ci, orig_idx in enumerate(idxs):
            results[orig_idx] = out[ci * N:(ci + 1) * N]

    return results


def compute_all_dz_hap(pops, dz_calls):
    """Fused Dz computation for all calls."""
    P = len(pops)
    N = pops[0].n.shape[0]
    F = _HapPopFlat(pops)
    pair_range = cp.arange(N, dtype=cp.int32)

    def expand(pop_arr):
        return (pop_arr[:, None] * N + pair_range[None, :]).ravel()

    groups = {k: [] for k in ['same', 'p1p2', 'p1p3', 'p2p3', 'diff']}
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

    group_results = {}

    for case in groups:
        calls = groups[case]
        if not calls:
            continue
        n_c = len(calls)
        M = n_c * N
        out = cp.empty(M, dtype=cp.float64)

        if case == 'diff':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
            fK = expand(cp.array([c[2] for c in calls], dtype=cp.int32))
            _launch(_DZ_DIFF_KERN,
                    (F.D, F.n, F.pA, F.qA, F.c1, F.c2, F.c3, F.c4,
                     fI, fJ, fK, out, M), M)
        elif case == 'p1p2':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[2] for c in calls], dtype=cp.int32))
            _launch(_DZ_IIJ_KERN,
                    (F.c1, F.c2, F.c3, F.c4, F.n, F.pA, F.qA,
                     fI, fJ, out, M), M)
        elif case == 'p1p3':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
            _launch(_DZ_IJI_KERN,
                    (F.c1, F.c2, F.c3, F.c4, F.n, F.pA, F.qA,
                     fI, fJ, out, M), M)
        elif case == 'p2p3':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
            _launch(_DZ_IJJ_KERN,
                    (F.c1, F.c2, F.c3, F.c4, F.n, F.pA, F.qA,
                     fI, fJ, out, M), M)
        elif case == 'same':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            _launch(_DZ_III_KERN,
                    (F.c1, F.c2, F.c3, F.c4, F.n, fI, out, M), M)

        group_results[case] = [out[ci * N:(ci + 1) * N] for ci in range(n_c)]

    output = [None] * len(dz_calls)
    for idx in range(len(dz_calls)):
        grp, pos = call_order[idx]
        output[idx] = group_results[grp][pos]
    return output


def compute_all_pi2_hap(pops, pi2_calls):
    """Fused pi2 computation for all calls."""
    P = len(pops)
    N = pops[0].n.shape[0]
    F = _HapPopFlat(pops)
    pair_range = cp.arange(N, dtype=cp.int32)

    def expand(pop_arr):
        return (pop_arr[:, None] * N + pair_range[None, :]).ravel()

    groups = {k: [] for k in
              ['same', 'triple', 'iikk', 'iikl', 'ijkk', 'ijij', 'shared', 'alldiff']}
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
            key = 'triple'
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

    group_results = {}

    for case in groups:
        calls = groups[case]
        if not calls:
            continue
        n_c = len(calls)
        M = n_c * N
        out = cp.empty(M, dtype=cp.float64)

        if case == 'alldiff':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
            fK = expand(cp.array([c[2] for c in calls], dtype=cp.int32))
            fL = expand(cp.array([c[3] for c in calls], dtype=cp.int32))
            _launch(_PI2_ALLDIFF_KERN,
                    (F.pA, F.qA, F.pB, F.qB, F.n, fI, fJ, fK, fL, out, M), M)

        elif case == 'iikk':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fK = expand(cp.array([c[2] for c in calls], dtype=cp.int32))
            _launch(_PI2_IIKK_KERN,
                    (F.pA, F.qA, F.pB, F.qB, F.n, fI, fK, out, M), M)

        elif case == 'iikl':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fK = expand(cp.array([c[2] for c in calls], dtype=cp.int32))
            fL = expand(cp.array([c[3] for c in calls], dtype=cp.int32))
            _launch(_PI2_IIKL_KERN,
                    (F.pA, F.qA, F.pB, F.qB, F.n, fI, fK, fL, out, M), M)

        elif case == 'ijkk':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
            fK = expand(cp.array([c[2] for c in calls], dtype=cp.int32))
            _launch(_PI2_IJKK_KERN,
                    (F.pA, F.qA, F.pB, F.qB, F.n, fI, fJ, fK, out, M), M)

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
                    (F.c1, F.c2, F.c3, F.c4, F.n, F.pA, F.qA, F.pB, F.qB,
                     fI, fJ, fK, out, M), M)

        elif case == 'triple':
            tp_l, sp_l = [], []
            for i, j, k, l in calls:
                cnt = {}
                for pp in (i, j, k, l):
                    cnt[pp] = cnt.get(pp, 0) + 1
                tp_l.append([pp for pp, c in cnt.items() if c == 3][0])
                sp_l.append([pp for pp, c in cnt.items() if c == 1][0])
            fI = expand(cp.array(tp_l, dtype=cp.int32))
            fJ = expand(cp.array(sp_l, dtype=cp.int32))
            _launch(_PI2_TRIPLE_KERN,
                    (F.c1, F.c2, F.c3, F.c4, F.n, F.pA, F.qA, F.pB, F.qB,
                     fI, fJ, out, M), M)

        elif case == 'ijij':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
            _launch(_PI2_IJIJ_KERN,
                    (F.c1, F.c2, F.c3, F.c4, F.n, F.pA, F.qA, F.pB, F.qB,
                     fI, fJ, out, M), M)

        elif case == 'same':
            fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
            _launch(_PI2_IIII_KERN,
                    (F.c1, F.c2, F.c3, F.c4, F.n, fI, out, M), M)

        group_results[case] = [out[ci * N:(ci + 1) * N] for ci in range(n_c)]

    output = [None] * len(pi2_calls)
    for idx in range(len(pi2_calls)):
        grp, pos = call_order[idx]
        output[idx] = group_results[grp][pos]
    return output


def compute_multi_pop_statistics_batch_hap(counts_per_pop, n_valid_per_pop,
                                           ld_statistics_module, stat_specs):
    """Compute all LD statistics using fused haplotype kernels.

    Drop-in replacement for _compute_multi_pop_statistics_batch.
    """
    from .ld_pipeline import PopData as _PopData

    n_pairs = counts_per_pop[0].shape[0]
    n_stats = len(stat_specs)

    pops = [_PopData(counts_per_pop[p], n_valid_per_pop[p])
            for p in range(len(counts_per_pop))]

    # Collect unique calls
    unique_calls = set()
    for _, calls in stat_specs:
        for _, stat_type, pop_indices in calls:
            unique_calls.add((stat_type, pop_indices))

    raw_cache = {}

    # DD
    dd_calls = [pi for st, pi in unique_calls if st == 'dd']
    if dd_calls:
        dd_results = compute_all_dd_hap(pops, dd_calls)
        for pi, val in zip(dd_calls, dd_results):
            raw_cache[('dd', pi)] = val

    # Dz
    dz_calls = [pi for st, pi in unique_calls if st == 'dz']
    if dz_calls:
        dz_results = compute_all_dz_hap(pops, dz_calls)
        for pi, val in zip(dz_calls, dz_results):
            raw_cache[('dz', pi)] = val

    # pi2
    pi2_calls = [pi for st, pi in unique_calls if st == 'pi2']
    if pi2_calls:
        pi2_results = compute_all_pi2_hap(pops, pi2_calls)
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
