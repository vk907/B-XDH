from estimator.lwe_parameters import LWEParameters as LWEest
from estimator import *

params = LWEest(
     n=768,
     q=12289,#1280,256 bit
     Xs=ND.CenteredBinomial(2),
     Xe=ND.CenteredBinomial(4),
     m=768,
     tag="example"
)
r=LWE.estimate(params)