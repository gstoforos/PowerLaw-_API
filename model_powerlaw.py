import numpy as np
from scipy.optimize import curve_fit

def power_law_model(gamma_dot, k, n):
    return k * np.power(gamma_dot, n)

def fit_powerlaw(data):
    shear_rates = np.array(data.get("shear_rates", []))
    shear_stresses = np.array(data.get("shear_stresses", []))

    flow_rate = float(data.get("flow_rate", 1))
    diameter = float(data.get("diameter", 1))
    density = float(data.get("density", 1))

    if len(shear_rates) != len(shear_stresses):
        return {"error": "Mismatched data lengths."}

    try:
        popt, _ = curve_fit(power_law_model, shear_rates, shear_stresses, maxfev=10000)
        k, n = popt
        predictions = power_law_model(shear_rates, k, n)
        residuals = shear_stresses - predictions
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((shear_stresses - np.mean(shear_stresses))**2)
        r_squared = 1 - (ss_res / ss_tot)

        mu_app = k * (np.mean(shear_rates) ** (n - 1))

        if flow_rate > 0 and diameter > 0 and density > 0:
            Q = flow_rate
            D = diameter
            rho = density
            Re = (8 * rho * Q) / (np.pi * D * mu_app)
        else:
            Re = None

        return {
            "model": "Power Law",
            "k": round(k, 6),
            "n": round(n, 6),
            "r_squared": round(r_squared, 6),
            "mu_app": round(mu_app, 6),
            "re": round(Re, 2) if Re is not None else None,
            "equation": f"τ = {k:.3g}·γ̇^{n:.3g}"
        }

    except Exception as e:
        return {"error": f"Fitting failed: {str(e)}"}

