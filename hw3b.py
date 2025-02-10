import scipy.integrate as spi
import scipy.special as sp
import numpy as np

def gamma_function(alpha):
    """Computes the Gamma function for a given alpha."""
    return sp.gamma(alpha)

def km_coefficient(m):
    """Computes the coefficient K_m for the t-distribution."""
    return gamma_function((m + 1) / 2) / (np.sqrt(m * np.pi) * gamma_function(m / 2))

def integrand(u, m):
    """Defines the integrand function for the probability computation."""
    return (1 + (u**2 / m))**(-(m + 1) / 2)

def compute_probability(z, m):
    """Computes the probability F(z) using numerical integration."""
    Km = km_coefficient(m)
    result, _ = spi.quad(integrand, -np.inf, z, args=(m))
    return Km * result

if __name__ == "__main__":
    for _ in range(3):
        m = int(input("Enter the degrees of freedom (7, 11, or 15): "))
        z = float(input("Enter the value of z: "))
        probability = compute_probability(z, m)
        print(f"F(z) for m={m}, z={z}: {probability:.4f}")
