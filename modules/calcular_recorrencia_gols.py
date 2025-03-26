import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson, nbinom
from collections import Counter

def calcular_recorrencia_gols(gols, golsMarcados, intervalo=False, intervalo_gols=(1, 2)):
    """
    Calculates the probability of a team scoring exactly 'gols' in past matches
    or within a goal range (if 'intervalo' is True).
    
    Returns:
        float: Probability of scoring the specified goal amount.
    """
    total_jogos = len(golsMarcados)
    if total_jogos == 0:
        return 0

    if intervalo:
        ocorrencias = sum(1 for g in golsMarcados if intervalo_gols[0] <= g <= intervalo_gols[1])
    else:
        ocorrencias = golsMarcados.count(gols)

    return ocorrencias / total_jogos

def calcular_distribuicao_gols(golsMarcados, plot=False):
    """
    Calculates the goal frequency distribution.
    
    Returns:
        dict: Goal frequency dictionary.
    """
    total_jogos = len(golsMarcados)
    if total_jogos == 0:
        return {}

    frequencias = Counter(golsMarcados)
    distribuicao = {g: f / total_jogos for g, f in frequencias.items()}

    # Optionally, plot distribution
    if plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(distribuicao.keys()), y=list(distribuicao.values()), color='blue', alpha=0.7)
        plt.title("Goal Distribution")
        plt.xlabel("Goals Scored")
        plt.ylabel("Probability")
        plt.grid(True)
        plt.show()

    return distribuicao

def calcular_variancia_e_desvio(golsMarcados):
    """
    Calculates variance & standard deviation for goal scoring volatility.
    
    Returns:
        tuple: (variance, standard deviation)
    """
    if len(golsMarcados) == 0:
        return 0, 0

    variancia = np.var(golsMarcados)
    desvio_padrao = np.std(golsMarcados)

    return variancia, desvio_padrao

def calcular_probabilidade_media(golsMarcados, tipo="média", intervalo_gols=(1, 2)):
    """
    Calculates goal probability based on historical averages or a goal range.
    
    Returns:
        float: Average goal probability.
    """
    if tipo == "média":
        return np.mean(golsMarcados) if golsMarcados else 0
    elif tipo == "intervalo":
        return calcular_recorrencia_gols(None, golsMarcados, intervalo=True, intervalo_gols=intervalo_gols)

    return 0

def calcular_probabilidade_ao_menos_x_gols(golsMarcados, x):
    """
    Calculates probability of scoring at least 'x' goals in a match.
    
    Returns:
        float: Probability of scoring at least 'x' goals.
    """
    total_jogos = len(golsMarcados)
    if total_jogos == 0:
        return 0

    jogos_ao_menos_x = sum(1 for g in golsMarcados if g >= x)
    return jogos_ao_menos_x / total_jogos

def calcular_distribuicao_acumulada(golsMarcados, max_gols):
    """
    Calculates cumulative probability of scoring up to 'max_gols'.
    
    Returns:
        float: Cumulative probability.
    """
    total_jogos = len(golsMarcados)
    if total_jogos == 0:
        return 0

    jogos_ate_max = sum(1 for g in golsMarcados if g <= max_gols)
    return jogos_ate_max / total_jogos

def prever_distribuicao_futura(golsMarcados, metodo="poisson", plot=False):
    """
    Predicts future goal distribution using Poisson or Negative Binomial Models.
    
    Returns:
        dict: Predicted probabilities for future goal outcomes.
    """
    if len(golsMarcados) == 0:
        return {}

    media_gols = np.mean(golsMarcados)

    if metodo == "poisson":
        distribuicao_pred = {g: poisson.pmf(g, media_gols) for g in range(10)}
    elif metodo == "nbinom":
        var_gols = np.var(golsMarcados)
        p = media_gols / var_gols if var_gols > media_gols else 0.5
        distribuicao_pred = {g: nbinom.pmf(g, 1, p) for g in range(10)}

    if plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(distribuicao_pred.keys()), y=list(distribuicao_pred.values()), color='red', alpha=0.7)
        plt.title(f"Predicted Goal Distribution ({metodo.capitalize()})")
        plt.xlabel("Goals")
        plt.ylabel("Probability")
        plt.grid(True)
        plt.show()

    return distribuicao_pred

def comparar_probabilidades_reais_preditas(golsMarcados, metodo="poisson"):
    """
    Compares actual goal distribution with predicted goal probabilities.
    
    Returns:
        dict: Comparison of real vs predicted probabilities.
    """
    real_dist = calcular_distribuicao_gols(golsMarcados)
    pred_dist = prever_distribuicao_futura(golsMarcados, metodo)

    comparacao = {g: {"Real": real_dist.get(g, 0), "Predito": pred_dist.get(g, 0)} for g in range(10)}

    # Plot comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(10)
    real_values = [comparacao[g]["Real"] for g in range(10)]
    pred_values = [comparacao[g]["Predito"] for g in range(10)]

    plt.bar(x - 0.2, real_values, width=0.4, label="Real", color="blue", alpha=0.7)
    plt.bar(x + 0.2, pred_values, width=0.4, label="Predicted", color="red", alpha=0.7)

    plt.xticks(x)
    plt.xlabel("Goals")
    plt.ylabel("Probability")
    plt.title(f"Real vs Predicted Goal Distribution ({metodo.capitalize()})")
    plt.legend()
    plt.grid(True)
    plt.show()

    return comparacao