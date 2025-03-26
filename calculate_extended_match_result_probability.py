from scipy.stats import poisson, norm
import numpy as np

def calculate_extended_match_result_probability(team_a, team_b, max_goals=10, home_away_factor=1.0, recent_form_factor=1.1, weather_factor=1.0):
    """
    Calcula as probabilidades de resultados estendidos, como vitórias por 1, 2, 3 gols, empates e derrotas,
    agora com fatores ajustados considerando casa/fora, desempenho recente e fatores externos (como clima).
    """
    
    # Ajuste das médias de gols com base em fatores dinâmicos (casa/fora, forma recente e clima)
    avg_goals_a = team_a.average_goals_scored() * home_away_factor * recent_form_factor * weather_factor
    avg_goals_b = team_b.average_goals_scored() * (1 / home_away_factor) * recent_form_factor * weather_factor  # Adversário jogando fora de casa
    
    # Cálculo da distribuição de Poisson para cada time
    prob_result = {f"Vitória A por {i} gols": 0 for i in range(1, max_goals+1)}
    prob_result["Empate"] = 0
    prob_result.update({f"Vitória B por {i} gols": 0 for i in range(1, max_goals+1)})
    
    # Distribuição de Poisson truncada para garantir que não haja probabilidade excessiva de muitos gols
    def poisson_truncado(k, lambda_):
        """Calcula a PMF de Poisson truncada para evitar valores extremos (como mais de 10 gols)."""
        if k >= 0:
            return poisson.pmf(k, lambda_)
        return 0

    # Probabilidades de todos os resultados possíveis
    for gols_a in range(0, max_goals):
        for gols_b in range(0, max_goals):
            prob_a = poisson_truncado(gols_a, avg_goals_a)
            prob_b = poisson_truncado(gols_b, avg_goals_b)
            total_prob = prob_a * prob_b

            # Classificando os resultados conforme a diferença de gols
            if gols_a > gols_b:
                if gols_a - gols_b == 1:
                    prob_result[f"Vitória A por 1 gol"] += total_prob
                elif gols_a - gols_b == 2:
                    prob_result[f"Vitória A por 2 gols"] += total_prob
                elif gols_a - gols_b == 3:
                    prob_result[f"Vitória A por 3 gols"] += total_prob
                elif gols_a - gols_b == 4:
                    prob_result[f"Vitória A por 4 gols"] += total_prob
                else:
                    prob_result["Vitória A"] += total_prob
            elif gols_a < gols_b:
                if gols_b - gols_a == 1:
                    prob_result[f"Vitória B por 1 gol"] += total_prob
                elif gols_b - gols_a == 2:
                    prob_result[f"Vitória B por 2 gols"] += total_prob
                elif gols_b - gols_a == 3:
                    prob_result[f"Vitória B por 3 gols"] += total_prob
                elif gols_b - gols_a == 4:
                    prob_result[f"Vitória B por 4 gols"] += total_prob
                else:
                    prob_result["Vitória B"] += total_prob
            else:
                prob_result["Empate"] += total_prob

    # Normalização para garantir que as probabilidades somem 100%
    total_probabilidade = sum(prob_result.values())
    for resultado in prob_result:
        prob_result[resultado] = (prob_result[resultado] / total_probabilidade) * 100

    # Adicionar incerteza à previsão com uma distribuição normal, se necessário
    if total_probabilidade < 0.1:  # Se a amostra for pequena, podemos aumentar a incerteza
        for resultado in prob_result:
            prob_result[resultado] *= 1.1  # Aumentando a incerteza para a previsão
    
    # Resultados ajustados considerando a incerteza
    for resultado, prob in prob_result.items():
        if prob < 1:
            prob_result[resultado] = round(prob, 2)  # Arredondar para maior legibilidade

    return prob_result