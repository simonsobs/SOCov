# What do we want this code to do:
# Overarching code to iterate over l1 and l2 to make the whole covmat
# smaller shell code that takes in derivatives and computes the matrix element for a 
# given l1, l2, and spectrum combination
# Other smaller codes will generate the derivatives through numerical (or analytical) means

import numpy as np
import matplotlib.pyplot as plt

import SOgen_cov_modules as gcm

labelXY='TT'
labelWZ='EE'
labelList = ['TT','EE','BB','TE']

ell_u,ell_l,specXY_u,specWZ_u,specXY_l,specWZ_l= gcm.get_CMB(labelXY,labelWZ,labelList)

# Default CAMB return is Dl not Cl
specXY_u = gcm.DltoCl(ell_u,specXY_u)
specXY_l = gcm.DltoCl(ell_l,specXY_l)
specWZ_u = gcm.DltoCl(ell_u,specWZ_u)
specWZ_l = gcm.DltoCl(ell_l,specWZ_l)

noiseXYorig, noiseWZorig = gcm.get_noise(labelXY,labelWZ,labelList)
covXYWZ_u  = gcm.get_cov_u(labelXY,labelWZ,labelList,fsky=0.35)

# # Making sure they are the right shape/length
# noiseXY = 0*np.ones(len(ell_u))
# noiseWZ = 0*np.ones(len(ell_u))

# noiseXY[:len(noiseXYorig)] = noiseXYorig
# noiseWZ[:len(noiseWZorig)] = noiseWZorig

# noiseXY = gcm.DltoCl(ell_u,noiseXY)
# noiseWZ = gcm.DltoCl(ell_u,noiseWZ)

fig, ax = plt.subplots(2,1)
ax[0].plot(gcm.CltoDl(ell_u,specXY_u), color='k')
ax[0].set_xlim([2,4000])
ax[0].set_ylim([10,6500])
topCl = gcm.CltoDl(ell_u,specXY_u+np.sqrt(np.diag(covXYWZ_u)))
botCl  = gcm.CltoDl(ell_u,specXY_u-np.sqrt(np.diag(covXYWZ_u)))
ax[0].fill_between(ell_u,botCl,topCl)
ax[0].plot(gcm.CltoDl(ell_u,specXY_u), color='k')

topCl = gcm.CltoDl(ell_u,specWZ_u+np.sqrt(np.diag(covXYWZ_u)))
botCl  = gcm.CltoDl(ell_u,specWZ_u-np.sqrt(np.diag(covXYWZ_u)))

ax[1].fill_between(ell_u,botCl,topCl)
ax[1].plot(gcm.CltoDl(ell_u,specWZ_u), color='k')

ax[1].plot(gcm.CltoDl(ell_u,specWZ_u), color='k')
ax[1].set_xlim([2,4000])
ax[1].set_ylim([0,60])



plt.legend()
#plt.yscale('log')

plt.savefig('huh.png')

# lmin1 = 2
# lmax1 = 20
# lmin2=2
# lmax2=20
# fsky = 0.35
# l1vec = np.arange(lmin1,lmax1)
# l2vec = np.arange(lmin2,lmax2)

# del12 = np.eye(len(l1vec),len(l2vec))

# ell,CTTu, CEEu, CBBu, CTEu, Cdd, CdTu, CdEu= np.loadtxt('testdat/planck_lensing_wp_highL_bestFit_20130627_lenspotentialCls.dat', unpack=True)
# # lens_potential_output_file is specified a file is output containing unlensed scalar (+tensor if calculated) spectra
# #  CX are l(l+1)Cl/2π, and d is the deflection angle, so Cdd=[l(l+1)]2ClΦ/2π, CdT=[l(l+1)]3/2ClΦT/2π, CdE=[l(l+1)]3/2ClΦE/2π. 

# NTT = CTTu
# NEE = CEEu
# NBB = CBBu
# NTE = CTEu
# Cpp = 2*np.pi*Cdd/(ell*(ell+1))
# ell, CTTl, CEEl, CBBl, CTEl = np.loadtxt('testdat/planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat',unpack=True)

# covtt = np.zeros((len(l1vec),len(l2vec)))

# #def 
# for l1 in l1vec:
#     for l2 in l2vec:
#         ind1 = l1-lmin1
#         ind2 = l2-lmin2
#         term1 = (1/(2*l1+1))*del12[ind1,ind2] *\
#         ((CTTl[ind1]+NTT[ind1])*(CTTl[ind2]+NTT[ind2]) +(CTTl[ind1]+NTT[ind1])*(CTTl[ind2]+NTT[ind2]))

#         term2 = 0

#         term3 = 0
#         covtt[ind1,ind2] = (1/fsky)*(term1 + term2 + term3)

# fig, ax = plt.subplots(1,1)

# xvals = np.linspace(lmin1,lmax1,10)
# ax.set_xticks([float(val) for val in xvals])
# xlabels=[str(int(val)) for val in xvals]

# yvals = np.linspace(lmin2,lmax2,10)

# ax.set_yticks([float(val) for val in yvals])
# ylabels=[str(int(val)) for val in yvals]


# img = ax.imshow(covtt)
# ax.set_xticklabels(xlabels)
# ax.set_yticklabels(ylabels)

# plt.savefig('covtt.png')


