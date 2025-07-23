import numpy as np
from scipy.optimize import curve_fit

def r2_score(y_true, y_pred):
    ss_res = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def fit_powerlaw(shear_rates, shear_stresses, flow_rate, diameter, density, re_critical=4000):
    # Power Law model: τ = k * γ̇^n
    def model(gamma_dot, k, n):
        return k * np.power(gamma_dot, n)

    popt, _ = curve_fit(model, shear_rates, shear_stresses, bounds=(0, np.inf))
    k, n = popt

    predicted = model(np.array(shear_rates), *popt)
    r2 = r2_score(shear_stresses, predicted)

    tau0 = 0.0
    avg_shear_rate = np.mean(shear_rates)
    mu_app = k * avg_shear_rate ** (n - 1)

    velocity = flow_rate / (np.pi * (diameter / 2) ** 2)
    re = (density * velocity ** (2 - n) * diameter ** n) / (k * 8 ** (n - 1))

    q_critical = (np.pi * diameter ** 2 / 4) * ((re_critical * k * 8 ** (n - 1)) / (density * diameter ** n)) ** (1 / (2 - n))

    equation = f"τ = {k:.3f}·γ̇^{n:.3f}"

    return {
        "equation": equation,
        "tau0": tau0,
        "k": k,
        "n": n,
        "r2": r2,
        "mu_app": mu_app,
        "re": re,
        "re_critical": re_critical,
        "q_critical": q_critical
    }
