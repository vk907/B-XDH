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
r=LWE.estimate(params)


