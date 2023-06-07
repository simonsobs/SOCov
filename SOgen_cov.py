# What do we want this code to do:
# Overarching code to iterate over l1 and l2 to make the whole covmat
# smaller shell code that takes in derivatives and computes the matrix element for a 
# given l1, l2, and spectrum combination
# Other smaller codes will generate the derivatives through numerical (or analytical) means

import numpy as np
import matplotlib.pyplot as plt
lmin1 = 2
lmax1 = 20
lmin2=2
lmax2=20
fsky = 0.35
l1vec = np.arange(lmin1,lmax1)
l2vec = np.arange(lmin2,lmax2)

del12 = np.eye(len(l1vec),len(l2vec))

ell,CTTu, CEEu, CBBu, CTEu, Cdd, CdTu, CdEu= np.loadtxt('testdat/planck_lensing_wp_highL_bestFit_20130627_lenspotentialCls.dat', unpack=True)
# lens_potential_output_file is specified a file is output containing unlensed scalar (+tensor if calculated) spectra
#  CX are l(l+1)Cl/2π, and d is the deflection angle, so Cdd=[l(l+1)]2ClΦ/2π, CdT=[l(l+1)]3/2ClΦT/2π, CdE=[l(l+1)]3/2ClΦE/2π. 

NTT = CTTu
NEE = CEEu
NBB = CBBu
NTE = CTEu
Cpp = 2*np.pi*Cdd/(ell*(ell+1))
ell, CTTl, CEEl, CBBl, CTEl = np.loadtxt('testdat/planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat',unpack=True)

covtt = np.zeros((len(l1vec),len(l2vec)))

#def 
for l1 in l1vec:
    for l2 in l2vec:
        ind1 = l1-lmin1
        ind2 = l2-lmin2
        term1 = (1/(2*l1+1))*del12[ind1,ind2] *\
        ((CTTl[ind1]+NTT[ind1])*(CTTl[ind2]+NTT[ind2]) +(CTTl[ind1]+NTT[ind1])*(CTTl[ind2]+NTT[ind2]))

        term2 = 0

        term3 = 0
        covtt[ind1,ind2] = (1/fsky)*(term1 + term2 + term3)

fig, ax = plt.subplots(1,1)

xvals = np.linspace(lmin1,lmax1,10)
ax.set_xticks([float(val) for val in xvals])
xlabels=[str(int(val)) for val in xvals]

yvals = np.linspace(lmin2,lmax2,10)

ax.set_yticks([float(val) for val in yvals])
ylabels=[str(int(val)) for val in yvals]


img = ax.imshow(covtt)
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)

plt.savefig('covtt.png')

# testl1 = np.arange(2,10)
# testl2 = np.arange(2,20)


# for testl1 in testl1:
#     for test2 in testl2:
#         print(testl1,test2,del12[testl1-lmin1,test2-lmin2])
