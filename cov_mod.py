# Code by Hidde Janse to compute covariance
# Modifications by Renee Hlozek to test CAMB deriv output

import numpy as np
import matplotlib.pyplot as plt
import camb
import tqdm
import sys



# Step 1. Create some example spectra
Nspec = 50
lmax_lensed = 3000
lmax_unlensed = 2500
lmax_pp = 2500

def symmetrize(xy):
	if xy == "et": return "te"
	if xy == "bt": return "tb"
	if xy == "be": return "eb"
	return xy

param_ranges = {
	"ombh2" : np.random.normal(0.0224, 0.001, Nspec),
	"omch2" : np.random.normal(0.120, 0.01, Nspec),
	"ns" : np.random.normal(0.965, 0.04, Nspec),
	"As" : 1e-10 * np.exp(np.random.normal(3.045, 0.16, Nspec)),
	"tau" : np.random.normal(0.054, 0.07, Nspec),
	"cosmomc_theta" : np.random.normal(104.11e-4, 3.0e-6, Nspec),
}

params_fiducial = {
	"ombh2" : 0.0224,
	"omch2" : 0.120,
	"ns" : 0.965,
	"As" : 1e-10 * np.exp(3.045),
	"tau" : 0.054,
	"cosmomc_theta" : 104.11e-4,
}

extra_args = {
	"lmax" : 6000,
	"lens_potential_accuracy" : 1
}

# Compute fiducial spectrum
#params = extra_args | params_fiducial

params={**extra_args, **params_fiducial}

pars = camb.set_params(**params)
res = camb.get_results(pars)

fiducial = {}

lfacpp = np.linspace(2,lmax_pp+1,1)

spectra = ["tt","ee","bb","te"]

ls = np.arange(2, lmax_lensed+1)
ls_u = np.arange(2, lmax_unlensed+1)

for i, nspec in enumerate(spectra):
	fiducial[nspec] = res.get_cmb_power_spectra(CMB_unit = "muK", raw_cl = True)["total"][2:lmax_lensed+1,i]
	fiducial[nspec + "u"] = res.get_cmb_power_spectra(CMB_unit = "muK", raw_cl = True)["unlensed_scalar"][2:lmax_unlensed+1,i]
	fiducial["l"+nspec] = ls*(ls+1)*res.get_cmb_power_spectra(CMB_unit = "muK", raw_cl = True)["total"][2:lmax_lensed+1,i]/(2*np.pi)
	fiducial["l"+nspec + "u"] = ls_u*(ls_u+1)*res.get_cmb_power_spectra(CMB_unit = "muK", raw_cl = True)["unlensed_scalar"][2:lmax_unlensed+1,i]/(2*np.pi)

fiducial["pp"] = res.get_lens_potential_cls(CMB_unit = "muK", raw_cl = True)[2:lmax_pp+1,0]
fiducial["phi"] = fiducial["pp"]/((lfacpp*(lfacpp+1))**2) 

specgen=False

if specgen==True:
	spec = np.zeros((lmax_lensed-1, 4, Nspec))
	lspec = np.zeros((lmax_lensed-1, 4, Nspec))
	spec_pp = np.zeros((lmax_pp-1, Nspec))
	spec_phi = np.zeros((lmax_pp-1, Nspec))
	spec_u = np.zeros((lmax_unlensed-1, 4, Nspec))
	lspec_u = np.zeros((lmax_unlensed-1, 4, Nspec))
	for n in tqdm.tqdm(range(Nspec)):
#		params = extra_args | { k : param_ranges[k][n] for k in param_ranges }
		tmpdict = {}
		for k in param_ranges:
			tmpdict.update({k:param_ranges[k][n]})
		params = {**extra_args, **tmpdict}
		
		pars = camb.set_params(**params)
		res = camb.get_results(pars)
		cmb_spec = res.get_cmb_power_spectra(CMB_unit = "muK", raw_cl = True)["total"]
		cmb_uspec = res.get_cmb_power_spectra(CMB_unit = "muK", raw_cl = True)["unlensed_scalar"]
		pp_spec = res.get_lens_potential_cls(CMB_unit = "muK", raw_cl = True)
		phi_spec = pp_spec/((lfacpp*(lfacpp+1))**2)

		spec[:,:,n] = cmb_spec[2:lmax_lensed+1,:]
		spec_u[:,:,n] = cmb_uspec[2:lmax_unlensed+1,:]
		lspec[:,:,n] = ls[:,None]*(ls[:,None]+1)*cmb_spec[2:lmax_lensed+1,:]/(2*np.pi)
		lspec_u[:,:,n] = ls_u[:,None]*(ls_u[:,None]+1)*cmb_uspec[2:lmax_unlensed+1,:]/(2*np.pi)
		spec_pp[:,n] = pp_spec[2:lmax_pp+1,0]
		spec_phi[:,n] = phi_spec[2:lmax_pp+1,0]

	np.savez("spectra.npz", spec = spec, lspec=lspec,specu = spec_u, lspecu=lspec_u,specpp = spec_pp,spec_phi=spec_phi, )
	
	sys.exit()
else:
	data = np.load("spectra.npz")
	spec = data["spec"]
	spec_u = data["specu"]
	lspec = data["lspec"]
	lspec_u = data["lspecu"]
	spec_pp = data["specpp"]
	spec_phi = data["spec_phi"]

means = {}
covs = {}

for i, nspec in tqdm.tqdm(enumerate(spectra), total = len(spectra)):
	means[nspec] = np.average(spec[:,i,:], axis = -1)
	means[nspec + "u"] = np.average(spec_u[:,i,:], axis = -1)
	means["l"+nspec] = np.average(spec[:,i,:], axis = -1)
	means["l"+nspec + "u"] = np.average(spec_u[:,i,:], axis = -1)

means["pp"] = np.average(spec_pp, axis = -1)
means["phi"] = np.average(spec_phi, axis = -1)

for i, nspec in enumerate(spectra):
	for j, mspec in enumerate(spectra):
		#covs[nspec + mspec] = np.average(spec[:,np.newaxis,i,:] * spec[np.newaxis,:,j,:], axis = -1) - means[nspec][:,np.newaxis] * means[mspec][np.newaxis,:]
		covs[nspec+mspec] = 0.0
		print(f"Cov {nspec}{mspec} done.")

ls = np.arange(2, lmax_unlensed+1)

for i, nspec in enumerate(spectra):
	for j, mspec in enumerate(spectra):
		# Cov[XY,ZW](l1,l2) ~ delta(l1,l2) / (2 l + 1) * ( Cl[XZ] * Cl[YW] + Cl[XW] * Cl[YZ] )
		x,y = nspec
		z,w = mspec
		
		xz = symmetrize(x+z)
		yw = symmetrize(y+w)
		xw = symmetrize(x+w)
		yz = symmetrize(y+z)
		
		print(f"Cov {nspec}{mspec} = Cl{xz}*Cl{yw} + Cl{xw}*Cl{yz}")
		
		ClXZ = np.zeros((len(ls), Nspec))
		ClYW = np.zeros((len(ls), Nspec))
		ClXW = np.zeros((len(ls), Nspec))
		ClYZ = np.zeros((len(ls), Nspec))
		
		if xz in spectra: ClXZ = spec_u[:, spectra.index(xz), :]
		if yw in spectra: ClYW = spec_u[:, spectra.index(yw), :]
		if xw in spectra: ClXW = spec_u[:, spectra.index(xw), :]
		if yz in spectra: ClYZ = spec_u[:, spectra.index(yz), :]
		
		covs[nspec+mspec+"u"] = np.diag( np.average(ClXZ * ClYW + ClXW * ClYZ, axis = -1) / (2.0 * ls + 1.0) )
		
		#covs[nspec + mspec + "u"] = np.average(spec_u[:,np.newaxis,i,:] * spec_u[np.newaxis,:,j,:], axis = -1) - means[nspec + "u"][:,np.newaxis] * means[mspec + "u"][np.newaxis,:]
		
		print(f"Cov {nspec}{mspec}u done.")

covs["pppp"] = np.diag( 2.0 * np.average(spec_pp ** 2.0, axis = -1) / (2.0 * ls + 1.0) )

#print(covs["tttt"].shape)
#print(covs["ttttu"].shape)
#print(covs["pppp"].shape)

fig, ax = plt.subplots(1, 1, figsize = (12, 8))

#fig.delaxes(axes[0,1])

ax.plot(np.diag(covs["ttttu"]), c = "C0", lw = 2)
ax.plot(np.diag(covs["eeeeu"]), c = "C1", lw = 2)
ax.plot(np.diag(covs["bbbbu"]), c = "C2", lw = 2)
ax.plot(np.diag(covs["teteu"]), c = "C3", lw = 2)

ax.semilogy()

plt.savefig("covs.pdf", bbox_inches = "tight")

# Step 2. Compute the gradient dCl/dClu
dCldClu = { k : np.zeros((lmax_lensed-1, lmax_unlensed-1)) for k in spectra }
dCldClpp = { k : np.zeros((lmax_lensed-1, lmax_pp-1)) for k in spectra }
dl2CldClphi = { k : np.zeros((lmax_lensed-1, lmax_pp-1)) for k in spectra }

mean = {}
var = {}

for i, nspec in enumerate(spectra):
	mean[nspec] = np.average(spec[:,i,:], axis = 1)[...,np.newaxis]
	mean[nspec + "u"] = np.average(spec_u[:,i,:], axis = 1)[...,np.newaxis]
	mean["l"+nspec] = np.average(lspec[:,i,:], axis = 1)[...,np.newaxis]
	mean["l"+nspec + "u"] = np.average(lspec_u[:,i,:], axis = 1)[...,np.newaxis]
	var[nspec] = np.var(spec[:,i,:], axis = 1)[np.newaxis,...]
	var[nspec + "u"] = np.var(spec_u[:,i,:], axis = 1)[np.newaxis,...]
	var["l"+nspec] = np.var(lspec[:,i,:], axis = 1)[np.newaxis,...]
	var["l"+nspec + "u"] = np.var(lspec_u[:,i,:], axis = 1)[np.newaxis,...]

mean["pp"] = np.average(spec_pp, axis = 1)[...,np.newaxis]
var["pp"] = np.var(spec_pp, axis = 1)[np.newaxis,...]

mean["phi"] = np.average(spec_phi, axis = 1)[...,np.newaxis]
var["phi"] = np.var(spec_phi, axis = 1)[np.newaxis,...]

for i, nspec in enumerate(spectra):
	for j in tqdm.tqdm(range(lmax_lensed-1), desc = nspec.upper()):
		dCldClu[nspec][j,:] = np.average( (spec[:,i,:] - mean[nspec])[np.newaxis,j,:] * (spec_u[:,i,:] - mean[nspec+"u"]), axis = 1 ) / var[nspec+"u"]
		dCldClpp[nspec][j,:] = np.average( (spec[:,i,:] - mean[nspec])[np.newaxis,j,:] * (spec_pp - mean["pp"]), axis = 1 ) / var["pp"]
		dl2CldClphi[nspec][j,:] = np.average( (lspec[:,i,:] - mean["l"+nspec])[np.newaxis,j,:] * (spec_phi - mean["phi"]), axis = 1 ) / var["phi"]

print("Covariances         : ", list(covs.keys()))
print("Fiducial spectra    : ", list(fiducial.keys()))
print("Unlensed derivatives: ", list(dCldClu.keys()))
print("Lensing derivatives : ", list(dCldClpp.keys()))
print("CAMB Lensing derivatives : ", list(dl2CldClphi.keys()))

#print(dCldClpp["tt"])

np.savez("matrices.npz",
	**{ "cov"+k:covs[k] for k in covs },
	**{ "fid"+k:fiducial[k] for k in fiducial },
	**{ "dCl" + k + "dCl" + k + "u" : dCldClu[k] for k in dCldClu },
	**{ "dCl" + k + "dClpp" : dCldClpp[k] for k in dCldClpp },
	**{ "dl2Cl" + k + "dClphi" : dl2CldClphi[k] for k in dl2CldClphi }
)
