#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:09:14 2021

Spin-Flip and Unrestricted: CIS and TDHF code

@author: julio
"""

import numpy as np
import psi4

np.set_printoptions(precision=6, linewidth=200, threshold=2000, suppress=True)
psi4.core.set_output_file('output.dat', True)
psi4.set_options({'basis': 'sto-6g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'reference':  'uhf'})

π = np.pi
α = 0.007297352
c = 1.0/α

## the SCF
mol = psi4.geometry(""" 
                    Be   0.0,  0.0,  0.260000 
                    H    0.0,  0.0, -1.040000
                    symmetry c1 
                    no_reorient """)
mol = psi4.geometry(""" 
                    Li   0.0,  0.0,  0.360000 
                    H    0.0,  0.0, -1.140000
                    symmetry c1 
                    no_reorient """)
E_uhf, Ψ = psi4.energy('SCF', return_wfn=True, mol=mol )

CA = np.asarray(Ψ.Ca())
CB = np.asarray(Ψ.Cb())
εA = np.asarray(Ψ.epsilon_a())
εB = np.asarray(Ψ.epsilon_b())
nA = Ψ.nalpha()
nB = Ψ.nbeta()

integrals = psi4.core.MintsHelper(Ψ.basisset())
I  = np.asarray(integrals.ao_eri())

# =============================================================================
# Preliminary blocks
# =============================================================================

IA_aiΥδ = 1.0*np.einsum('Aa, Bi, ABCD -> aiCD', CA[:, nA:], CA[:, :nA], I, optimize=True)
IB_aiΥδ = 1.0*np.einsum('Aa, Bi, ABCD -> aiCD', CB[:, nB:], CB[:, :nB], I, optimize=True)
KA_aiΥδ = 1.0*np.einsum('Aa, Bi, ADCB -> aiCD', CA[:, nA:], CA[:, :nA], I, optimize=True)
KB_aiΥδ = 1.0*np.einsum('Aa, Bi, ADCB -> aiCD', CB[:, nB:], CB[:, :nB], I, optimize=True)

# =============================================================================
# A_UCIS Block
# =============================================================================

A_AAAA = 1.0*np.einsum("Cj, Db, aiCD -> aijb", CA[:, :nA], CA[:, nA:], IA_aiΥδ, optimize=True) - 1.0*np.einsum("Cj, Db, aiCD -> aijb", CA[:, :nA], CA[:, nA:], KA_aiΥδ, optimize=True)
A_AABB = 1.0*np.einsum("Cj, Db, aiCD -> aijb", CB[:, :nB], CB[:, nB:], IA_aiΥδ, optimize=True)
A_BBAA = 1.0*np.einsum("Cj, Db, aiCD -> aijb", CA[:, :nA], CA[:, nA:], IB_aiΥδ, optimize=True)
A_BBBB = 1.0*np.einsum("Cj, Db, aiCD -> aijb", CB[:, :nB], CB[:, nB:], IB_aiΥδ, optimize=True) - 1.0*np.einsum("Cj, Db, aiCD -> aijb", CB[:, :nB], CB[:, nB:], KB_aiΥδ, optimize=True)

FF_AA_mo  = np.einsum('ab, ij -> iajb', np.diag( εA[nA:]), np.eye( len( εA[:nA]) ))
FF_AA_mo -= np.einsum('ij, ab -> iajb', np.diag( εA[:nA]), np.eye( len( εA[nA:]) ))

FF_BB_mo  = np.einsum('ab, ij -> iajb', np.diag( εB[nB:]), np.eye( len( εB[:nB]) ))
FF_BB_mo -= np.einsum('ij, ab -> iajb', np.diag( εB[:nB]), np.eye( len( εB[nB:]) ))


A_AA = A_AAAA.swapaxes(0, 1) + FF_AA_mo
A_AB = A_AABB.swapaxes(0, 1)
A_BA = A_BBAA.swapaxes(0, 1)
A_BB = A_BBBB.swapaxes(0, 1) + FF_BB_mo

A_AA = A_AA.reshape( len(εA[nA:])*len(εA[:nA]), len(εA[nA:])*len(εA[:nA]) )
A_AB = A_AB.reshape( ( len(εA[nA:])*len(εA[:nA]), len(εB[nB:])*len(εB[:nB]) ) )
A_BA = A_BA.reshape( ( len(εB[nB:])*len(εB[:nB]), len(εA[nA:])*len(εA[:nA]) ) )
A_BB = A_BB.reshape( len(εB[nB:])*len(εB[:nB]), len(εB[nB:])*len(εB[:nB]) )

A_UCIS = np.asarray( np.bmat([[A_AA, A_AB],[A_BA, A_BB]]) )

# =============================================================================
# B_UTDHF Block
# =============================================================================

B_AAAA = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", CA[:, :nA], CA[:, nA:], IA_aiΥδ, optimize=True) - 1.0*np.einsum("Cb, Dj, aiCD -> aibj", CA[:, nA:], CA[:, :nA], KA_aiΥδ, optimize=True)
B_AABB = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", CB[:, :nB], CB[:, nB:], IA_aiΥδ, optimize=True) 
B_BBAA = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", CA[:, :nA], CA[:, nA:], IB_aiΥδ, optimize=True)
B_BBBB = 1.0*np.einsum("Cb, Dj, aiCD -> aijb", CB[:, :nB], CB[:, nB:], IB_aiΥδ, optimize=True) - 1.0*np.einsum("Cb, Dj, aiCD -> aibj", CB[:, nB:], CB[:, :nB], KB_aiΥδ, optimize=True)

B_AAAA = (B_AAAA.swapaxes(0, 1)).swapaxes(2, 3) # iajb
B_AABB = (B_AABB.swapaxes(0, 1)).swapaxes(2, 3)
B_BBAA = (B_BBAA.swapaxes(0, 1)).swapaxes(2, 3)
B_BBBB = (B_BBBB.swapaxes(0, 1)).swapaxes(2, 3)

B_AAAA = B_AAAA.reshape( len(εA[nA:])*len(εA[:nA]), len(εA[nA:])*len(εA[:nA]) )
B_AABB = B_AABB.reshape( ( len(εA[nA:])*len(εA[:nA]), len(εB[nB:])*len(εB[:nB]) ) )
B_BBAA = B_BBAA.reshape( ( len(εB[nB:])*len(εB[:nB]), len(εA[nA:])*len(εA[:nA]) ) )
B_BBBB = B_BBBB.reshape( len(εB[nB:])*len(εB[:nB]), len(εB[nB:])*len(εB[:nB]) )

B_UTDHF = np.asarray( np.bmat([[B_AAAA,B_AABB],[B_BBAA,B_BBBB]]) )

# =============================================================================
# A_SF Blocks (as is)
# =============================================================================

A_ABAB = -1.*np.einsum("Aa, Bi, Cj, Db, ADCB -> aijb", CB[:, nB:], CA[:, :nA], CA[:, :nA], CB[:, nB:], I, optimize=True)
A_BABA = -1.*np.einsum("Aa, Bi, Cj, Db, ADCB -> aijb", CA[:, nA:], CB[:, :nB], CB[:, :nB], CA[:, nA:], I, optimize=True)

## Fock
A_ABAB += np.einsum("ab, ij -> aijb", np.diag( εB[nB:] ), np.eye( len( εA[:nA] ) ))
A_ABAB -= np.einsum("ij, ab -> aijb", np.diag( εA[:nA] ), np.eye( len( εB[nB:] ) ))

A_BABA += np.einsum("ab, ij -> aijb", np.diag( εA[nA:] ), np.eye( len( εB[:nB] ) ))
A_BABA -= np.einsum("ij, ab -> aijb", np.diag( εB[:nB] ), np.eye( len( εA[nA:] ) ))

A_ABAB = A_ABAB.swapaxes(0, 1)
A_BABA = A_BABA.swapaxes(0, 1)

A_ABAB = A_ABAB.reshape( ( len(εA[:nA])*len(εB[nB:]), len(εA[:nA])*len(εB[nB:]) ) )
A_BABA = A_BABA.reshape( ( len(εA[nA:])*len(εB[:nB]), len(εA[nA:])*len(εB[:nB]) ) )

A_SF = np.asarray( np.bmat([[ A_ABAB, np.zeros(( A_ABAB.shape[1], A_BABA.shape[1] ))], [np.zeros(( A_BABA.shape[0], A_ABAB.shape[1] )),  A_BABA]]))

# =============================================================================
# B_SF Blocks (as is)
# =============================================================================

B_BAAB  = -1.*np.einsum("Aa, Bi, Cb, Dj, ADCB -> aibj", CA[:, nA:], CB[:, :nB], CB[:, nB:], CA[:, :nA], I, optimize=True)
B_ABBA  = -1.*np.einsum("Aa, Bi, Cb, Dj, ADCB -> aibj", CB[:, nB:], CA[:, :nA], CA[:, nA:], CB[:, :nB], I, optimize=True)

B_BAAB = B_BAAB.reshape( ( len(εA[nA:])*len(εB[:nB]), len(εB[nB:])*len(εA[:nA]) ) , order="F")
B_ABBA = B_ABBA.reshape( ( len(εB[nB:])*len(εA[:nA]), len(εA[nA:])*len(εB[:nB]) ) , order="F")

B_SF = np.asarray( np.bmat([[ np.zeros(( B_ABBA.shape[0], B_BAAB.shape[1] )), B_ABBA], [B_BAAB,  np.zeros(( B_BAAB.shape[0], B_ABBA.shape[1] ))]]))

# =============================================================================
# Solve UCIS equations
# =============================================================================

ω_UCIS, t_UCIS = np.linalg.eigh(A_UCIS)

print("EUHF = %4.8f" % E_uhf)
print("UCIS     : " + str(27.211386245988*ω_UCIS[:12]) )

# =============================================================================
# Solve SF-CIS equations
# =============================================================================

ω_sfCISp, t_sfCISp = np.linalg.eigh(A_ABAB)
ω_sfCISm, t_sfCISm = np.linalg.eigh(A_BABA)

print("SF-UCIS+ : " + str(27.211386245988*ω_sfCISp) )
print("SF-UCIS- : " + str(27.211386245988*ω_sfCISm) )

# =============================================================================
# Solve UTDHF equations
# =============================================================================

ω_tdhfR, t_tdhfR = np.linalg.eig(  ( (A_UCIS - B_UTDHF ) @ (A_UCIS + B_UTDHF ) )  )
ω_tdhfL, t_tdhfL = np.linalg.eig(  ( (A_UCIS + B_UTDHF ) @ (A_UCIS - B_UTDHF ) )  )

print( "U-TDHF-L : " + str(27.211386245988*np.sqrt(np.sort(ω_tdhfL.real))) )
print( "U-TDHF-R : " + str(27.211386245988*np.sqrt(np.sort(ω_tdhfR.real))) )

# =============================================================================
# Solve SFTDHF equations
# =============================================================================

Cbig = np.asarray( np.bmat( [[ A_SF, B_SF], [-B_SF, -A_SF]] ))

ωC_utdhf, t_utdhf = np.linalg.eig(Cbig)

print( "SF-TDHF  : " + str(27.211386245988*np.sort(ωC_utdhf)) )
