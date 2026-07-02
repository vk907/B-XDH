from estimator.lwe_parameters import LWEParameters as LWEest
from estimator import *
import math
etas=4
etae=4
sig=2*math.sqrt(etae/2)
Q=18433
N=512

params = LWEest(
     n=N,
     q=Q,#1280,256 bit
     Xs=ND.CenteredBinomial(etas), # s
     Xe=ND.DiscreteGaussian(sig), # 2e
     m=N,
     tag="example"
)
r=LWE.estimate(params)#classical

"""
bkw                  :: rop: ≈2^180.9, m: ≈2^166.0, mem: ≈2^160.1, b: 11, t1: 0, t2: 19, ℓ: 10, #cod: 453, #top: 3, #test: 56, tag: coded-bkw
usvp                 :: rop: ≈2^132.3, red: ≈2^132.3, δ: 1.004250, β: 364, d: 1020, tag: usvp
bdd                  :: rop: ≈2^129.2, red: ≈2^128.1, svp: ≈2^128.2, β: 349, η: 383, d: 1015, tag: bdd
dual                 :: rop: ≈2^137.3, mem: ≈2^88.1, m: 512, β: 379, d: 1024, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^129.1, red: ≈2^129.0, guess: ≈2^125.5, β: 349, p: 6, ζ: 10, t: 30, β': 355, N: ≈2^73.5, m: 512
"""

params = LWEest(
     n=N,
     q=Q,#1280,256 bit
     Xs=ND.DiscreteGaussian(sig), # s0+s1
     Xe=ND.DiscreteGaussian(2*sig), # 2e0+2e1
     m=N,
     tag="example"
)
r=LWE.estimate(params)#classical for prekis

