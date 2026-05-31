#!/usr/bin/env python3
"""
Calibração e Espessura Analysis using ROOT methods
Replicates the analysis from beta/11_calibracao&espessura.ipynb using ROOT fitting
"""

import ROOT
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.integrate import quad
import re

# ROOT configuration (same as cal_lib.py)
ROOT.gROOT.SetBatch(True)
ROOT.TH1D.AddDirectory(False)

# ============================================================================
# Helper Functions from cal_lib.py
# ============================================================================

def get_centroid(canais, contagens, idx):
    """Get centroid and error using ROOT TH1D (from cal_lib.py)"""
    h = ROOT.TH1D(f"h_{idx}", "", 1024, 0, 1024)
    for c, n in zip(canais, contagens):
        h.SetBinContent(int(c) + 1, float(n))
    
    centroide = h.GetMean()
    erro = h.GetMeanError()
    return centroide, erro


# ============================================================================
# Additional Helper Functions from Notebook
# ============================================================================

def load_spectrum(path):
    """Load spectrum data from ASCII file"""
    text = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()
    channels = []
    counts = []
    
    for line in text:
        match = re.match(r'\s*(\d+),\s*(\d+)', line)
        if match:
            channels.append(float(match.group(1)))
            counts.append(float(match.group(2)))
    
    return np.array(channels), np.array(counts)


def compute_chi2(o, e, dof=None):
    """Compute chi-squared"""
    mask = e > 0
    chi2 = np.sum(((o[mask] - e[mask]) ** 2) / e[mask])
    if dof is not None:
        return chi2 / dof
    return chi2


def compute_stat(canais, contagens, model=None):
    """Compute statistics on spectrum data"""
    c = np.array(canais)
    N = np.array(contagens)
    
    N_t = np.sum(N)
    if N_t == 0:
        return 0, 0, None
    
    # Direct calculation (no fitting model)
    mu = np.sum(c * N) / N_t
    sigma = np.sqrt(np.sum((c - mu) ** 2 * N)) / N_t
    return mu, sigma, float('nan')


def linear(x, m, b):
    """Linear function"""
    return m * x + b


def fit_linear_with_root(x, y, sigma_y):
    """
    Fit a line using ROOT TGraphErrors and TF1
    Returns: (m, b, m_err, b_err, chi2, ndf)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sigma_y = np.asarray(sigma_y, dtype=np.float64)
    
    # Create ROOT graph with errors
    gr = ROOT.TGraphErrors(len(x), x, y, np.zeros(len(x)), sigma_y)
    
    # Create linear function
    fit = ROOT.TF1("f", "[0] + [1]*x", float(np.min(x)), float(np.max(x)))
    fit.SetLineColor(ROOT.kRed)
    
    # Fit (S Q = silent, quiet)
    gr.Fit(fit, "S Q")
    
    m = fit.GetParameter(1)
    b = fit.GetParameter(0)
    m_err = fit.GetParError(1)
    b_err = fit.GetParError(0)
    chi2 = fit.GetChisquare()
    ndf = fit.GetNDF()
    
    return m, b, m_err, b_err, chi2, ndf


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("=" * 70)
    print("CALIBRAÇÃO E ESPESSURA ANALYSIS (ROOT-based)")
    print("=" * 70)
    
    # Change to beta directory for file access
    import os
    os.chdir("beta")
    
    # ========================================================================
    # 1. ANÁLISE BI-207
    # ========================================================================
    print("\n--- Análise Bi-207 ---")
    channels_bi, counts_bi = load_spectrum('BIPR3.ASC')
    
    # Define ROIs for Bi-207 peaks
    roi_data = {
        '1K': (98, 105),
        '1L': (115, 123),
        '2K': (207, 213),
        '2L': (224, 232)
    }
    
    bi_results = {}
    for roi_name, (min_ch, max_ch) in roi_data.items():
        roi = slice(min_ch, max_ch)
        mu, sigma, chi2 = compute_stat(channels_bi[roi], counts_bi[roi], model=None)
        bi_results[roi_name] = {'mu': mu, 'sigma': sigma, 'chi2': chi2}
        print(f"{roi_name}: μ = {mu:.4f} ± {sigma:.4f}; χ² = {chi2:.4f}")
    
    # ========================================================================
    # 2. ANÁLISE PULSER
    # ========================================================================
    print("\n--- Análise Pontos do Pulser ---")
    pulser_files = ['PULSER02.ASC', 'PULSER04.ASC', 'PULSER06.ASC', 
                    'PULSER08.ASC', 'PULSER10.ASC', 'PULSER12.ASC']
    pulser_dials = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    pulser_results = {}
    
    pulser_C = []
    pulser_C_sigma = []
    
    for i, (dial, fname) in enumerate(zip(pulser_dials, pulser_files)):
        channels, counts = load_spectrum(fname)
        
        # Use different ROIs for different pulser files
        roi_config = {
            'PULSER02.ASC': (0, 1200),
            'PULSER04.ASC': (0, 1200),
            'PULSER06.ASC': (123, 135),
            'PULSER08.ASC': (0, 1200),
            'PULSER10.ASC': (1, 222),
            'PULSER12.ASC': (0, 1200),
        }
        
        min_ch, max_ch = roi_config.get(fname, (0, 1200))
        roi = slice(min_ch, max_ch)
        mu, sigma, chi2 = compute_stat(channels[roi], counts[roi], model=None)
        
        pulser_results[fname] = {'mu': mu, 'sigma': sigma, 'chi2': chi2}
        pulser_C.append(mu)
        pulser_C_sigma.append(sigma)
        print(f"{fname} (dial {dial}): μ = {mu:.4f} ± {sigma:.4f}; χ² = {chi2:.4f}")
    
    pulser_C = np.array(pulser_C)
    pulser_C_sigma = np.sqrt(np.array(pulser_C_sigma) ** 2 + 0.5 ** 2)
    
    # ========================================================================
    # 3. CALIBRAÇÃO: DIALS → CANAIS (using ROOT)
    # ========================================================================
    print("\n--- Calibração: Dials → Canais ---")
    m_p, b_p, m_p_err, b_p_err, chi2_p, ndf_p = fit_linear_with_root(pulser_dials, pulser_C, pulser_C_sigma)
    
    print(f"m_p = {m_p:.6f} ± {m_p_err:.6f} canais/dial")
    print(f"b_p = {b_p:.2f} ± {b_p_err:.2f} canais")
    chi2_red = chi2_p / ndf_p if ndf_p > 0 else 0
    print(f"χ²_red = {chi2_red:.4f}")
    
    # ========================================================================
    # 4. CALIBRAÇÃO: CANAIS → ENERGIA
    # ========================================================================
    print("\n--- Calibração: Canais → Energia ---")
    
    # Bi-207 known energies
    bi_conv_enes = np.array([0.4817, 0.5538, 0.9757, 1.0478])
    bi_conv_mus = np.array([bi_results['1K']['mu'], bi_results['1L']['mu'], 
                            bi_results['2K']['mu'], bi_results['2L']['mu']])
    bi_conv_sigmas = np.array([bi_results['1K']['sigma'], bi_results['1L']['sigma'],
                               bi_results['2K']['sigma'], bi_results['2L']['sigma']])
    
    # Convert Bi channels to dials
    bi_conv_dials = (bi_conv_mus - b_p) / m_p
    bi_conv_dials_sigma = np.sqrt((bi_conv_sigmas / m_p) ** 2 + 
                                  ((bi_conv_mus - b_p) * m_p_err / (m_p ** 2)) ** 2 + 
                                  (b_p_err / m_p) ** 2)
    
    # Fit Bi-207 points to get kappa (using ROOT)
    kappa, kappa_b, kappa_err, kappa_b_err, chi2_bi, ndf_bi = fit_linear_with_root(
        bi_conv_dials, bi_conv_enes, bi_conv_dials_sigma
    )
    
    print(f"κ = {kappa:.6f} ± {kappa_err:.6f} MeV/dial")
    
    # Energy calibration coefficients
    m = kappa / m_p
    m_err = np.sqrt((m_p_err * kappa / (m_p ** 2)) ** 2 + (kappa_err / m_p) ** 2)
    b = -b_p * kappa / m_p
    b_err = np.sqrt((b_p_err * kappa / m_p) ** 2 + 
                    (b_p * kappa_err / m_p) ** 2 + 
                    (b_p * kappa * m_p_err / (m_p ** 2)) ** 2)
    
    print(f"m = {m:.10f} ± {m_err:.10f} MeV/ch")
    print(f"b = {b:.10f} ± {b_err:.10f} MeV")
    
    # ========================================================================
    # 5. ANÁLISE CS-137
    # ========================================================================
    print("\n--- Análise Cs-137 ---")
    cs_channels, cs_counts = load_spectrum('CPR1.ASC')
    
    cs_conv_min, cs_conv_max = 100, 210
    cs_conv_roi = slice(cs_conv_min, cs_conv_max)
    cs_conv_mu_ch, cs_conv_sigma_ch, cs_conv_chi2 = compute_stat(
        cs_channels[cs_conv_roi], cs_counts[cs_conv_roi], model=None
    )
    
    cs_conv_mu = m * cs_conv_mu_ch + b
    cs_conv_sigma = np.sqrt((m_err * cs_conv_mu_ch) ** 2 + 
                             (m * cs_conv_sigma_ch) ** 2 + 
                             b_err ** 2)
    
    print(f"Cs-137: μ = {cs_conv_mu:.6f} ± {cs_conv_sigma:.6f} MeV; χ² = {cs_conv_chi2:.4f}")
    
    # ========================================================================
    # 6. ANÁLISE DE ESPESSURA
    # ========================================================================
    print("\n--- Análise de Espessura ---")
    
    cs_exp = cs_conv_mu
    cs_exp_sigma = cs_conv_sigma
    cs_theo = 0.624215  # Cs-137 theoretical peak energy
    rho = 9.40000e-01  # g/cm³ (density)
    
    # Stopping power data
    energy = np.array([
        1.000E-02, 1.250E-02, 1.500E-02, 1.750E-02, 2.000E-02, 2.500E-02, 
        3.000E-02, 3.500E-02, 4.000E-02, 4.500E-02, 5.000E-02, 5.500E-02, 
        6.000E-02, 7.000E-02, 8.000E-02, 9.000E-02, 1.000E-01, 1.250E-01, 
        1.500E-01, 1.750E-01, 2.000E-01, 2.500E-01, 3.000E-01, 3.500E-01, 
        4.000E-01, 4.500E-01, 5.000E-01, 5.500E-01, 6.000E-01, 7.000E-01, 
        8.000E-01, 9.000E-01, 1.000E+00, 1.250E+00, 1.500E+00, 1.750E+00, 
        2.000E+00, 2.500E+00, 3.000E+00, 3.500E+00, 4.000E+00, 4.500E+00, 
        5.000E+00, 5.500E+00, 6.000E+00, 7.000E+00, 8.000E+00, 9.000E+00, 
        1.000E+01, 1.250E+01, 1.500E+01, 1.750E+01, 2.000E+01, 2.500E+01, 
        3.000E+01, 3.500E+01, 4.000E+01, 4.500E+01, 5.000E+01, 5.500E+01, 
        6.000E+01, 7.000E+01, 8.000E+01, 9.000E+01, 1.000E+02, 1.250E+02, 
        1.500E+02, 1.750E+02, 2.000E+02, 2.500E+02, 3.000E+02, 3.500E+02, 
        4.000E+02, 4.500E+02, 5.000E+02, 5.500E+02, 6.000E+02, 7.000E+02, 
        8.000E+02, 9.000E+02, 1.000E+03
    ])
    
    stopping_power = np.array([
        2.442E+01, 2.049E+01, 1.775E+01, 1.573E+01, 1.417E+01, 1.192E+01, 
        1.036E+01, 9.209E+00, 8.328E+00, 7.630E+00, 7.062E+00, 6.591E+00, 
        6.194E+00, 5.560E+00, 5.077E+00, 4.695E+00, 4.387E+00, 3.825E+00, 
        3.446E+00, 3.174E+00, 2.970E+00, 2.687E+00, 2.501E+00, 2.373E+00, 
        2.277E+00, 2.204E+00, 2.147E+00, 2.103E+00, 2.068E+00, 2.015E+00, 
        1.980E+00, 1.956E+00, 1.939E+00, 1.917E+00, 1.910E+00, 1.911E+00, 
        1.916E+00, 1.932E+00, 1.950E+00, 1.970E+00, 1.989E+00, 2.008E+00, 
        2.026E+00, 2.044E+00, 2.061E+00, 2.094E+00, 2.126E+00, 2.156E+00, 
        2.184E+00, 2.253E+00, 2.317E+00, 2.379E+00, 2.439E+00, 2.556E+00, 
        2.670E+00, 2.782E+00, 2.893E+00, 3.004E+00, 3.114E+00, 3.223E+00, 
        3.333E+00, 3.551E+00, 3.769E+00, 3.987E+00, 4.205E+00, 4.749E+00, 
        5.294E+00, 5.839E+00, 6.385E+00, 7.479E+00, 8.574E+00, 9.671E+00, 
        1.077E+01, 1.187E+01, 1.297E+01, 1.407E+01, 1.517E+01, 1.737E+01, 
        1.958E+01, 2.179E+01, 2.400E+01
    ])
    
    S = stopping_power * rho
    stopping_power_interp = interp1d(energy, S, kind='cubic', fill_value='extrapolate')
    
    # Calculate thickness
    def delta_x(E1, E2):
        integrand = lambda E: 1 / stopping_power_interp(E)
        result, error = quad(integrand, E1, E2)
        return result
    
    def delta_x_with_error(E1, E2, sigma_E1, sigma_E2):
        dx = delta_x(E1, E2)
        dS1 = stopping_power_interp(E1)
        dS2 = stopping_power_interp(E2)
        sigma_dx = np.sqrt((sigma_E1 / dS1) ** 2 + (sigma_E2 / dS2) ** 2)
        return dx, sigma_dx
    
    dx, sigma_dx = delta_x_with_error(cs_exp, cs_theo, cs_exp_sigma, 0)
    print(f"Δx = {dx:.6f} ± {sigma_dx:.6f} cm")
    
    # CSDA Range method
    csda_range = np.array([
        2.308E-04, 3.430E-04, 4.745E-04, 6.244E-04, 7.921E-04, 1.179E-03, 
        1.630E-03, 2.143E-03, 2.715E-03, 3.343E-03, 4.025E-03, 4.758E-03, 
        5.541E-03, 7.249E-03, 9.134E-03, 1.118E-02, 1.339E-02, 1.952E-02, 
        2.642E-02, 3.399E-02, 4.215E-02, 5.991E-02, 7.924E-02, 9.979E-02, 
        1.213E-01, 1.437E-01, 1.667E-01, 1.902E-01, 2.142E-01, 2.632E-01, 
        3.133E-01, 3.641E-01, 4.155E-01, 5.452E-01, 6.759E-01, 8.068E-01, 
        9.375E-01, 1.197E+00, 1.455E+00, 1.710E+00, 1.963E+00, 2.213E+00, 
        2.461E+00, 2.707E+00, 2.950E+00, 3.431E+00, 3.905E+00, 4.372E+00, 
        4.833E+00, 5.960E+00, 7.054E+00, 8.119E+00, 9.157E+00, 1.116E+01, 
        1.307E+01, 1.491E+01, 1.667E+01, 1.837E+01, 2.000E+01, 2.158E+01, 
        2.310E+01, 2.601E+01, 2.874E+01, 3.132E+01, 3.377E+01, 3.936E+01, 
        4.434E+01, 4.883E+01, 5.293E+01, 6.015E+01, 6.639E+01, 7.188E+01, 
        7.678E+01, 8.120E+01, 8.523E+01, 8.893E+01, 9.235E+01, 9.851E+01, 
        1.039E+02, 1.088E+02, 1.131E+02
    ])
    
    csda = csda_range * rho
    csda_interp = interp1d(energy, csda, kind='cubic', fill_value='extrapolate')
    
    deltax_csda = csda_interp(cs_theo) - csda_interp(cs_exp)
    sigma_deltax_csda = np.sqrt(
        (csda_interp(cs_exp + cs_exp_sigma) - csda_interp(cs_exp - cs_exp_sigma)) ** 2 + 
        (csda_interp(cs_theo + 0.01) - csda_interp(cs_theo - 0.01)) ** 2
    ) / 2
    
    print(f"Δx (CSDA) = {deltax_csda:.6f} ± {sigma_deltax_csda:.6f} cm")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
