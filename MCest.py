import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from typing import Tuple, Dict, List
import pandas as pd

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True

class EpidemicBranchingProcess:
    
    def __init__(self, R0: float, vaccine_efficacy: float):
        self.R0 = R0
        self.vaccine_efficacy = vaccine_efficacy
    
    def simulate_trajectory(self, v: float, threshold: int = 50) -> Tuple[bool, int, int]:
        R_eff = self.R0 * (1 - v * self.vaccine_efficacy)

        current_gen = 1
        total_cases = 1
        I1 = 0  
        I2 = 0  
        generation = 0
        
        while current_gen > 0 and total_cases < threshold:
    
            new_infections = np.random.poisson(R_eff * current_gen)
            
            if generation == 0:
                I1 = new_infections
            elif generation == 1:
                I2 = new_infections
            
            current_gen = new_infections
            total_cases += new_infections
            generation += 1
            
            if current_gen == 0:
                break
        
        epidemic_occurred = total_cases >= threshold
        return epidemic_occurred, I1, I2
    
    def estimate_epidemic_probability_naive(self, v: float, M: int = 10000, 
                                          threshold: int = 50) -> Tuple[float, float]:

        epidemics = []
        
        for _ in range(M):
            epidemic, _, _ = self.simulate_trajectory(v, threshold)
            epidemics.append(epidemic)
        
        epidemics = np.array(epidemics)
        p_hat = np.mean(epidemics)
        variance = np.var(epidemics) / M  # Variance de l'estimateur
        
        return p_hat, variance
    
    def estimate_epidemic_probability_control(self, v: float, M: int = 10000, 
                                            threshold: int = 50) -> Tuple[float, float, float]:

        R_eff = self.R0 * (1 - v * self.vaccine_efficacy)
        
        E_h = R_eff + R_eff**2
        
        # Simulation
        Y_values = []  
        h_values = []  
        
        for _ in range(M):
            epidemic, I1, I2 = self.simulate_trajectory(v, threshold)
            Y_values.append(float(epidemic))
            h_values.append(I1 + I2)
        
        Y_values = np.array(Y_values)
        h_values = np.array(h_values)
        
        cov_Yh = np.cov(Y_values, h_values)[0, 1]
        var_h = np.var(h_values)
        
        if var_h > 0:
            b = cov_Yh / var_h
        else:
            b = 0
        
        # Estimateur avec variable de contrôle
        Y_corrected = Y_values - b * (h_values - E_h)
        p_hat_vc = np.mean(Y_corrected)
        variance_vc = np.var(Y_corrected) / M
        
        return p_hat_vc, variance_vc, b
    
    def find_v_star(self, alpha: float = 0.05, tolerance: float = 0.001, 
                    M: int = 10000, threshold: int = 50) -> Tuple[float, float]:
        v_min, v_max = 0.0, 1.0
        p_min, _, _ = self.estimate_epidemic_probability_control(v_min, M//10, threshold)
        p_max, _, _ = self.estimate_epidemic_probability_control(v_max, M//10, threshold)
        
        if p_min < alpha:
            return 0.0, p_min
        if p_max > alpha:
            return 1.0, p_max

        while v_max - v_min > tolerance:
            v_mid = (v_min + v_max) / 2
            p_mid, _, _ = self.estimate_epidemic_probability_control(v_mid, M, threshold)
            
            if p_mid > alpha:
                v_min = v_mid
            else:
                v_max = v_mid
        
        v_star = (v_min + v_max) / 2
        p_star, _, _ = self.estimate_epidemic_probability_control(v_star, M, threshold)
        
        return v_star, p_star
    
    def plot_epidemic_probability_curve(self, v_values: np.ndarray, alpha: float = 0.05,
                                      M: int = 10000, threshold: int = 50):
        """
        Trace la courbe p(v) avec intervalles de confiance
        """
        p_naive = []
        p_control = []
        var_naive = []
        var_control = []
        
        for v in v_values:
            # Estimation naïve
            p_n, var_n = self.estimate_epidemic_probability_naive(v, M, threshold)
            p_naive.append(p_n)
            var_naive.append(var_n)
            
            # Estimation avec contrôle
            p_c, var_c, _ = self.estimate_epidemic_probability_control(v, M, threshold)
            p_control.append(p_c)
            var_control.append(var_c)
        
        p_naive = np.array(p_naive)
        p_control = np.array(p_control)
        std_naive = np.sqrt(var_naive)
        std_control = np.sqrt(var_control)
        
    
        plt.figure(figsize=(12, 8))
     
        plt.plot(v_values, p_naive, 'b-', label='Estimateur naïf', linewidth=2)
        plt.fill_between(v_values, p_naive - 1.96*std_naive, p_naive + 1.96*std_naive,
                        alpha=0.3, color='blue')
        
        plt.plot(v_values, p_control, 'r-', label='Avec variable de contrôle', linewidth=2)
        plt.fill_between(v_values, p_control - 1.96*std_control, p_control + 1.96*std_control,
                        alpha=0.3, color='red')
        
   
        plt.axhline(y=alpha, color='green', linestyle='--', label=f'Seuil α = {alpha}')
        
   
        plt.xlabel('Fraction vaccinée (v)')
        plt.ylabel('Probabilité d\'épidémie majeure')
        plt.title(f'Probabilité d\'épidémie majeure en fonction de la couverture vaccinale\n' + 
                 f'(R₀={self.R0}, e={self.vaccine_efficacy}, T={threshold})')
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, max(1.1, max(p_naive)*1.1))
        
        return p_naive, p_control, var_naive, var_control


def analyze_variance_reduction(process: EpidemicBranchingProcess, v_values: List[float],
                             M: int = 10000, threshold: int = 50) -> pd.DataFrame:
    """
    Analyse de l'efficacité relative de la réduction de variance
    """
    results = []
    
    for v in v_values:
        # Estimations
        p_n, var_n = process.estimate_epidemic_probability_naive(v, M, threshold)
        p_c, var_c, b = process.estimate_epidemic_probability_control(v, M, threshold)
        
        # Efficacité relative
        efficiency = var_n / var_c if var_c > 0 else np.inf
        
        results.append({
            'v': v,
            'p_naive': p_n,
            'p_control': p_c,
            'var_naive': var_n,
            'var_control': var_c,
            'coefficient_b': b,
            'efficacite_relative': efficiency,
            'reduction_variance_%': (1 - var_c/var_n) * 100 if var_n > 0 else 0
        })
    
    return pd.DataFrame(results)


def compute_v_star(R0: float, e: float, alpha: float, N: int, T: int, M: int) -> Dict:

    process = EpidemicBranchingProcess(R0, e)

    v_star, p_star = process.find_v_star(alpha, tolerance=0.001, M=M, threshold=T)

    N_vac = int(np.ceil(v_star * N))

    v_c = (1 - 1/R0) / e if R0 > 1 else 0
    
    results = {
        'v_star': v_star,
        'N_vac': N_vac,
        'p_v_star': p_star,
        'v_collective': v_c,
        'R0': R0,
        'e': e,
        'alpha': alpha,
        'N': N,
        'T': T,
        'M': M
    }
    
    return results

if __name__ == "__main__":

    R0 = 3.8
    e = 0.8
    alpha = 0.05
    T = 121
    N = 100000
    M = 10000

    results = compute_v_star(R0, e, alpha, N, T, M)
    print(f"- v* = {results['v_star']:.3f} ({results['v_star']*100:.1f}%)")
    print(f"- Nombre à vacciner : {results['N_vac']} sur {N}")
    print(f"- Probabilité d'épidémie à v* : {results['p_v_star']:.4f}")

    process = EpidemicBranchingProcess(R0, e)
    v_values = np.linspace(0, 1, 20)
    p_naive, p_control, var_naive, var_control = process.plot_epidemic_probability_curve(
        v_values, alpha, M//2, T)

    plt.axvline(x=results['v_star'], color='red', linestyle=':', linewidth=2,
                label=f"v* = {results['v_star']:.3f}")
    plt.scatter([results['v_star']], [results['p_v_star']], color='red', s=100, zorder=5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    v_test = [0.2, 0.3, 0.4, 0.5]
    df_variance = analyze_variance_reduction(process, v_test, M//2, T)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(df_variance['v'], df_variance['var_naive'], 'b-o', 
                 label='Variance estimateur naïf', linewidth=2, markersize=8)
    plt.semilogy(df_variance['v'], df_variance['var_control'], 'r-s', 
                 label='Variance avec contrôle', linewidth=2, markersize=8)
    plt.xlabel('Fraction vaccinée (v)')
    plt.ylabel('Variance de l\'estimateur (échelle log)')
    plt.title('Réduction de variance par la méthode de contrôle')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()