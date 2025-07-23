import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def fit_powerlaw(shear_rates, shear_stresses, flow_rate, diameter, density, re_critical=4000):
    # Power Law model: τ = k * γ̇^n
    def model(gamma_dot, k, n):
        return k * np.power(gamma_dot, n)

    popt, _ = curve_fit(model, shear_rates, shear_stresses, bounds=(0, np.inf))
    k, n = popt

    predicted = model(np.array(shear_rates), *popt)
    r2 = r2_score(shear_stresses, predicted)

    # tau0 for Power Law is always 0
    tau0 = 0.0

    # Apparent viscosity at average shear rate
    avg_shear_rate = np.mean(shear_rates)
    mu_app = k * avg_shear_rate ** (n - 1)

    # Calculate velocity and Re
    velocity = flow_rate / (np.pi * (diameter / 2) ** 2)
    re = (density * velocity ** (2 - n) * diameter ** n) / (k * 8 ** (n - 1))

    # Calculate q_critical: flow rate below which Re < Re_critical
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
