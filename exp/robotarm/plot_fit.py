import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# Use DejaVu Serif font
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.unicode_minus"] = False

# State dimensions and empirical log-log slopes
d = np.array([4, 6, 8], dtype=float)

uniform_slope = np.array([-0.2806, -0.2094, -0.1662])
adv_slope = np.array([-0.3535, -0.2701, -0.2224])

# Fit the magnitude of the negative slope:
# |slope(d)| ≈ 1 / (a d^b + c)
def mag_model(d, a, b, c):
    return 1.0 / (a * d**b + c)

def slope_model(d, a, b, c):
    return -mag_model(d, a, b, c)

uniform_params, _ = curve_fit(
    mag_model,
    d,
    np.abs(uniform_slope),
    p0=(0.5, 1.0, 1.0),
    maxfev=10000,
)

adv_params, _ = curve_fit(
    mag_model,
    d,
    np.abs(adv_slope),
    p0=(0.8, 1.0, 0.5),
    maxfev=10000,
)

print("Uniform parameters:     a, b, c =", uniform_params)
print("Adversarial parameters: a, b, c =", adv_params)

# Smooth fitted curves
d_grid = np.linspace(4, 8, 200)
uniform_fit = slope_model(d_grid, *uniform_params)
adv_fit = slope_model(d_grid, *adv_params)

fig, ax = plt.subplots(figsize=(4.8, 3.4))

ax.scatter(d, uniform_slope, marker="o", label="Uniform data")
ax.plot(d_grid, uniform_fit, label="Uniform fit")

ax.scatter(d, adv_slope, marker="s", label="Adversarial data")
ax.plot(d_grid, adv_fit, label="Adversarial fit")

ax.set_xlabel("State dimension")
ax.set_ylabel("Empirical log-log slope")
ax.set_title("Dimension-dependent slope fitting")
ax.legend()
ax.grid(True, alpha=0.3)

out_dir = Path("CoRL_2026/fig")
out_dir.mkdir(parents=True, exist_ok=True)

fig.tight_layout()
fig.savefig(out_dir / "robotarm_slope_fit.png", dpi=300)
fig.savefig(out_dir / "robotarm_slope_fit.pdf")
plt.show()
