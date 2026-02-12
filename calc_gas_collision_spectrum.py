import numpy as np
from scipy.special import erf

import sys
sys.path.append('../')
import analysis_utils as utils

kb = 1.380649e-23
c = 299792458    # m / s
SI2ev = (1 / 1.6e-19) * c
ev2SI = 1 / (SI2ev)

T_gas = 293

def fmb(dp, mg, vbar):
    return np.exp(-1 * dp**2 / (8 * mg**2 * vbar**2)) / np.sqrt(2 * np.pi * vbar**2)

def xi(x):
    return np.sqrt(np.pi) * x * (1 - 2/(x**2)) * erf(x/2) * np.exp(-x**2 / 8) + 2 * np.exp(-3 * x**2 / 8)

def dgamma_dp_specular(sphere_radius, pp_kev, mg_amu, p_mbar, T_gas):
    A = 4 * np.pi * (sphere_radius)**2  # Surface area of sphere (m^2)

    pp = pp_kev * 1000 * ev2SI
    mg = mg_amu * 1.660538921e-27
    p_pascal = p_mbar * 100

    ng = p_pascal / (kb * T_gas)
    vbar = np.sqrt(kb * T_gas / mg)

    rate = (ng * A * pp / (4 * mg**2)) * fmb(pp, mg, vbar)
    rate_hz_kev = rate * 1000 * ev2SI
    return rate_hz_kev

def dgamma_dp_diffuse(sphere_radius, pp_kev, mg_amu, p_mbar, T_gas):
    A = 4 * np.pi * (sphere_radius)**2  # Surface area of sphere (m^2)

    pp = pp_kev * 1000 * ev2SI
    mg = mg_amu * 1.660538921e-27
    p_pascal = p_mbar * 100

    ng = p_pascal / (kb * T_gas)
    vbar = np.sqrt(kb * T_gas / mg)

    rate = (ng * A * pp / (4 * mg**2)) * fmb(pp, mg, vbar) * xi(pp/(mg*vbar))
    rate_hz_kev = rate * 1000 * ev2SI
    return rate_hz_kev

def zeta(a, c, d):
    prefactor = c**2 * d**2

    first_pre = 2 * a * np.sqrt(c**2 + d**2)
    first_0 = first_pre * d**2 * np.exp(-0.5 * a**2 / d**2)
    first_1 = first_pre * c**2 * np.exp(-0.5 * a**2 / c**2)

    second_pre = c * d * (a**2 - c**2 - d**2) * np.exp(-0.5 * a**2 / (c**2 + d**2)) * np.sqrt(2 * np.pi)
    second = erf(a * c / (np.sqrt(2) * d * np.sqrt(c**2 + d**2))) + erf(a * d / (np.sqrt(2) * c * np.sqrt(c**2 + d**2)))

    denominator = 2 * (c**2 + d**2)**(5/2)

    return prefactor * ((first_0 + first_1) + second_pre * second) / denominator
    
def dgamma_dp_diffuse_noneq(sphere_radius, pp_kev, mg_amu, p_mbar, tl=293, th=300):
    A = 4 * np.pi * (sphere_radius)**2  # Surface area of sphere (m^2)

    pp = pp_kev * 1000 * ev2SI      # N s
    mg = mg_amu * 1.660538921e-27   # kg
    
    p_pascal = p_mbar * 100
    ng = p_pascal / (kb * tl)

    vl = np.sqrt(kb * tl / mg)
    vh = np.sqrt(kb * th / mg)

    rate = ng * A * (1 / np.sqrt(2*np.pi)) * (1 / (vl*vh**2)) * zeta(pp/mg, vl, vh) / mg
    rate_hz_kev = rate / (SI2ev / 1000)
    
    return rate_hz_kev

def dgamma_dp_tot_fixedtemp(pp_kev, mg_amu, p_mbar, alpha, T_gas=293, sphere_radius=50e-9):
    rate_specular = dgamma_dp_specular(sphere_radius, pp_kev, mg_amu, p_mbar, T_gas)
    rate_diffuse  = dgamma_dp_diffuse(sphere_radius, pp_kev, mg_amu, p_mbar, T_gas)

    return (1 - alpha) * rate_specular + alpha * rate_diffuse

def dgamma_dp_tot_noneq(pp_kev, mg_amu, p_mbar, alpha, T_gas=293, T_sensor=900, sphere_radius=50e-9):
    rate_specular = dgamma_dp_specular(sphere_radius, pp_kev, mg_amu, p_mbar, T_gas)
    rate_diffuse_noneq = dgamma_dp_diffuse_noneq(sphere_radius, pp_kev, mg_amu, p_mbar, tl=T_gas, th=T_sensor)

    return (1 - alpha) * rate_specular + alpha * rate_diffuse_noneq

def get_drdqz(qq, drdq):
    drdq_iso = drdq / (4 * np.pi * qq**2)

    ret = np.empty_like(qq)
    for i, q in enumerate(qq):
        xx = qq[qq >= q]
        integrand = drdq_iso[qq >= q]

        # Another factor of two because we want rate
        # for both +z and -z
        ret[i] = 2 * 2 * np.pi * np.trapz(integrand*xx, xx)

    return qq, ret

def smear_drdqz_gauss(qq, drdqz, sigma_kev):
    """Convolve spectrum with a Gaussian kernel"""
    dq = qq[1] - qq[0]
    qq_gauss = np.arange(-2000, 2000, dq)
    gauss_kernel = utils.gauss(qq_gauss, A=1, mu=0, sigma=sigma_kev)

    # Pad the array to minimize edge effect
    # to get the rising tail when q -> 0
    # then pad with mirror image
    pad_len = gauss_kernel.size
    if qq[0] >= dq:
        padded_drdqz = np.pad(drdqz, (pad_len, 0), mode='symmetric')
    else:
        padded_drdqz = np.pad(drdqz, (pad_len, 0), mode='reflect')
    padded_drdqz = np.pad(padded_drdqz, (0, pad_len), mode='constant', constant_values=0)

    convolved = np.convolve(padded_drdqz, gauss_kernel, mode='valid')

    idx_start = (convolved.size - drdqz.size) // 2
    ret = convolved[idx_start : idx_start + drdqz.size] / np.sum(gauss_kernel)

    return qq, ret

if __name__ == '__main__':
    sphere_radius = 50e-9
    mg_amu = 131.293
    p_mbar = 1e-8
    T_gas = 293

    alpha_list = np.linspace(0, 1, 100)
    ts_list = np.linspace(294, 1000, 100)

    bins = np.arange(0, 2000, 25)
    qq_kev = 0.5 * (bins[1:] + bins[:-1])
    # qq_kev = np.linspace(10, 2500, 200)
    drdqz_all = np.empty(shape=(alpha_list.size, ts_list.size, qq_kev.size), dtype=np.float64)
    for i, alpha in enumerate(alpha_list):
        # print(i)
        for j, ts in enumerate(ts_list):
            drdq = dgamma_dp_tot_noneq(qq_kev, mg_amu, p_mbar, alpha, T_gas=T_gas, T_sensor=ts, sphere_radius=sphere_radius)
            qq, drdqz = get_drdqz(qq_kev, drdq)

            drdqz_all[i][j] = drdqz
    
    outdir = r'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/gas_signal'
    np.savez(f'{outdir}/xe_signal_{sphere_radius}nm_{p_mbar}mbar.npz', pressure=p_mbar, alpha_list=alpha_list, ts_list=ts_list, qq_kev=qq_kev, drdqz=drdqz_all)

