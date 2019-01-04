# Conventions
if "w" is present = real frequency axis else matsubara frequencies

zn_vec is all of the matsubara frequencies (a vector)
w_vec is all of the real frequency grid (a vector)


sEvec_c = self-energy array in the cluster basis (a vector of self-eneries, one for each matsubara frequency)
sEvec_ir = self-energy array in the irreducible representation
gfvec_c = green function array in the cluster basis
gfarr_ir = green function array in the irreducible basis
gfvec_cw = green function array in the cluster basis and real frequency axis
gfvec_irw = green function array in the cluster basis and real frequency axis

# t = tabular form (complex numbers) without the frequency grid (real or matsubara)
# to = tabular-out form with the frequency grid for writing to disk.
# when the green function is in a tabular or tabular-out form, the "vec" is not there
# because it is not a vector but a tabular form for mainly writing to disk among other things.

gf_ctow = green function array in the cluster basis tabular-out and real frequency axis
gf_cto = green function array in the cluster basis tabular-out and matsubara frequency

gf_ctw = green function array in the cluster basis tabular (complex) and real frequency axis
gf_ct = green function array in the cluster basis tabular (complex) and matsubara frequencies
gf_irt = green function array in the irreducible basis in tabular form and matsuabara frequencies
gf_irtw = green function array in the irreducible basis in tabular form and real frequencies


