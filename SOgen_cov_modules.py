import numpy as np
def CMB_Phi_Cov(labelXY,labelWZ,labelList,specPP,covPP):
    ''' using equation 5 from https://arxiv.org/pdf/2111.15036.pdf
    This computes the covariance including off-diagonal terms induced from
    lensing-CMB correlations.
    
    It takes in the labels of the covmat you want (e.g. TT), 
    the total list of possible labels we are considering, and the phi spectrum
    and covariance, which are computed/provided elsewhere.'''

    # Current issue: how to check if lensed/unlensed spectra are different lengths
    ell_u,ell_l,specXY_u,specWZ_u,specXY_l,specWZ_l=get_CMB(labelXY,labelWZ,labelList)
    
    # Getting the noise spectra
    noiseXYorig,noiseWZorig = get_noise(labelXY,labelWZ,labelList)

    # making the noise spectra as long as the lensed spectra
    noiseXY = 0*np.ones(len(ell_u))
    noiseWZ =0*np.ones(len(ell_u))
    noiseXY[0:len(noiseXYorig)] = noiseXYorig
    noiseWZ[0:len(noiseWZorig)] = noiseWZorig

    noiseXY = DltoCl(ell_u,noiseXY)
    noiseWZ = DltoCl(ell_u,noiseWZ)

    dspecXY_ldspecXY_u = get_deriv(specXY_l, specXY_u, numflag=True)
    dspecWZ_ldspecWZ_u = get_deriv(specWZ_l, specWZ_u, numflag=True)

    dspecXY_ldspecPP = get_deriv(specXY_l, specPP, numflag=True)
    dspecWZ_ldspecPP = get_deriv(specWZ_l, specPP, numflag=True)

    # Computing the diagonal term I


    # Computing the off-diagonal term II


    # computing the off-diagonal term III
    covXYWZ_u = 1

    covXYWZ_l = covXYWZ_u
# need to check that covPP is the same size as needed
    return covXYWZ_l

def get_deriv(specA, specB, numflag=True):
    #import numpy as np

    if numflag: return np.gradient(specA,specB)


def get_CMB(labelXY,labelWZ,labelList):
    #import numpy as np

    ell_l, CTTl, CEEl, CBBl, CTEl = \
        np.loadtxt('testdat/planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat',unpack=True)
    ell_u,CTTu, CEEu, CBBu, CTEu, Cdd, CdTu, CdEu= \
        np.loadtxt('testdat/planck_lensing_wp_highL_bestFit_20130627_lenspotentialCls.dat', unpack=True)
# lens_potential_output_file is specified a file is output containing unlensed scalar (+tensor if calculated) spectra
#  CX are l(l+1)Cl/2π, and d is the deflection angle, so Cdd=[l(l+1)]2ClΦ/2π, CdT=[l(l+1)]3/2ClΦT/2π, CdE=[l(l+1)]3/2ClΦE/2π. 


    inXY = labelList.index(labelXY)
    inWZ = labelList.index(labelWZ)

    specXY_u = locals()['C'+labelList[inXY]+'u'] 
    specXY_l = locals()['C'+labelList[inXY]+'l'] 
    specWZ_u = locals()['C'+labelList[inWZ]+'u'] 
    specWZ_l = locals()['C'+labelList[inWZ]+'l'] 

    return ell_u,ell_l,specXY_u,specWZ_u,specXY_l,specWZ_l
    

def get_noise(labelXY,labelWZ,labelList):
    #import numpy as np

    inXY = labelList.index(labelXY)
    inWZ = labelList.index(labelWZ)
    
    ell,NTT,NEE = np.loadtxt('testdat/Nell_EC_TPdep0_ACT19_fsky035.txt',unpack=True)
    
    NTT = 2*np.pi*NTT/(ell**2)
    NEE = 2*np.pi*NEE/(ell**2)
    # Check if they have overlapping items
    if set(labelXY) == set(labelWZ):
        # eg. TT and TT
        noiseXY = locals()['N'+labelList[inXY]]
        noiseWZ = locals()['N'+labelList[inWZ]]
    elif len(set(labelXY)) > len(set(labelWZ)):
        # eg TE TT
        print(labelXY,labelWZ, 'inside set check >')
        noiseXY = 0*locals()['N'+labelList[inXY]] # no cross term noise 
        noiseWZ = locals()['N'+labelList[inWZ]] # only need main noise for one leg
    elif len(set(labelXY)) < len(set(labelWZ)):
        # eg TT TE
        print(labelXY,labelWZ, 'inside set check <')
        noiseXY = locals()['N'+labelList[inXY]] # no cross term noise 
        noiseWZ = 0*locals()['N'+labelList[inWZ]] # only need main noise for one leg
    else: 
        # eg TT EE
        print(labelXY,labelWZ, 'inside set check else')
        noiseXY = 0*locals()['N'+labelList[inXY]] # no cross term noise 
        noiseWZ = 0*locals()['N'+labelList[inWZ]] # no cross term noise

    return noiseXY,noiseWZ

def get_cov_u(labelXY,labelWZ,labelList,fsky):
    #import numpy as np

    inXY = labelList.index(labelXY)
    inWZ = labelList.index(labelWZ)

    ell_u,ell_l,specXY_u,specWZ_u,specXY_l,specWZ_l=get_CMB(labelXY,labelWZ,labelList)
    norm_fac = 2./(fsky*(2*ell_u+1))

    specXY_u = DltoCl(ell_u,specXY_u)
    specXY_l = DltoCl(ell_l,specXY_l)
    specWZ_u = DltoCl(ell_u,specWZ_u)
    specWZ_l = DltoCl(ell_l,specWZ_l)

    noiseXYorig,noiseWZorig = get_noise(labelXY,labelWZ,labelList)
    print(noiseXYorig, 'inside get noise')
    # Making sure they are the right shape/length
    noiseXY = 0*np.ones(len(ell_u))
    noiseWZ =0*np.ones(len(ell_u))

    noiseXY[0:len(noiseXYorig)] = noiseXYorig
    noiseWZ[0:len(noiseWZorig)] = noiseWZorig

    noiseXY = DltoCl(ell_u,noiseXY)
    noiseWZ = DltoCl(ell_u,noiseWZ)

    print(f"{noiseXY[0:len(noiseXYorig)]} is the {labelXY} noise")
    print(f"{noiseWZ[0:len(noiseWZorig)]} is the {labelWZ} noise")

    covXYWZ_u = np.diag((specXY_u + noiseXY)*(specWZ_u + noiseWZ))

    covXYWZ_u[:,:]*=norm_fac

    return covXYWZ_u    

def DltoCl(ell,Dl):

    #import numpy as np
    Cl = Dl*(2*np.pi)/(ell*(ell+1))

    return Cl

def CltoDl(ell,Cl):

    #import numpy as np
    Dl = Cl*(ell*(ell+1))/(2*np.pi)

    return Dl

def get_phiCov(specPP):
    
    # This will be provided in the sacc file, don't need to recalculate here
    # an issue right now is how to ensure it is the right size etc., so currently sending in specPP

    covPP = np.diag(specPP)




