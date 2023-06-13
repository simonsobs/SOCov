
def CMB_Phi_Cov(labelXY,labelWZ,labelList,specPP,covPP):
        
    ell_u,ell_l,specXY_u,specWZ_u,specXY_l,specWZ_l=get_CMB(labelXY,labelWZ,labelList)
    
    # noiseXY,noiseWZ = get_noise(labelXY,labelWZ)
    # covXYWZ_u = get_cov_CMBu(labelXY,labelWZ)

    covXYWZ_u = 1

    covXYWZ_l = covXYWZ_u
# need to check that covPP is the same size as needed
    return covXYWZ_l

def get_CMB(labelXY,labelWZ,labelList):
    import numpy as np

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
    import numpy as np

    inXY = labelList.index(labelXY)
    inWZ = labelList.index(labelWZ)
    
    ell,NTT,NEE = np.loadtxt('testdat/Nell_EC_TPdep0_ACT19_fsky035.txt',unpack=True)
    
    NTT = 2*np.pi*NTT/(ell**2)
    NEE = 2*np.pi*NEE/(ell**2)
    # Check if they have overlapping items
    if set(labelXY) == set(labelWZ):
        noiseXY = locals()['N'+labelList[inXY]]
        noiseWZ = locals()['N'+labelList[inWZ]]
    elif set(labelXY) > set(labelWZ):
        noiseXY = locals()['N'+labelList[inXY]] # no cross term noise -- only need o
        noiseWZ = 0*locals()['N'+labelList[inWZ]] # no cross term noise
    else:
        noiseXY = 0*locals()['N'+labelList[inXY]] # no cross term noise -- only need o
        noiseWZ = locals()['N'+labelList[inWZ]] # no cross term noise


    return noiseXY,noiseWZ

def get_cov_u(labelXY,labelWZ,labelList,fsky):
    import numpy as np

    inXY = labelList.index(labelXY)
    inWZ = labelList.index(labelWZ)

    ell_u,ell_l,specXY_u,specWZ_u,specXY_l,specWZ_l=get_CMB(labelXY,labelWZ,labelList)
    norm_fac = 2./(fsky*(2*ell_u+1))

    specXY_u = DltoCl(ell_u,specXY_u)
    specXY_l = DltoCl(ell_l,specXY_l)
    specWZ_u = DltoCl(ell_u,specWZ_u)
    specWZ_l = DltoCl(ell_l,specWZ_l)

    noiseXYorig,noiseWZorig = get_noise(labelXY,labelWZ,labelList)

    # Making sure they are the right shape/length
    noiseXY = 0*np.ones(len(ell_u))
    noiseWZ =0*np.ones(len(ell_u))

    noiseXY[0:len(noiseXYorig)] = noiseXYorig
    noiseWZ[0:len(noiseWZorig)] = noiseWZorig

    noiseXY = DltoCl(ell_u,noiseXY)
    noiseWZ = DltoCl(ell_u,noiseWZ)

    print(noiseXY[0:len(noiseXYorig)])

    covXYWZ_u = np.diag((specXY_u + noiseXY)*(specWZ_u+noiseWZ))

    covXYWZ_u[:,:]*=norm_fac

    return covXYWZ_u    

def DltoCl(ell,Dl):

    import numpy as np
    Cl = Dl*(2*np.pi)/(ell*(ell+1))

    return Cl

def CltoDl(ell,Cl):

    import numpy as np
    Dl = Cl*(ell*(ell+1))/(2*np.pi)

    return Dl