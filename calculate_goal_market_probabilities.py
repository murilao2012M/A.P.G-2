from scipy.stats import poisson, norm
import numpy as np

def calculate_goal_market_probabilities(team_a, team_b, home_away_factor=1.0, performance_factor=1.0, competition_factor=1.0):
    """
    Calcula as probabilidades para os mercados de gols (over/under) considerando o desempenho dos times,
    fatores contextuais, e outros ajustes baseados no tipo de competição e no histórico recente.
    """
    # Média de gols dos times ajustada por fatores de casa/fora, desempenho e tipo de competição
    avg_goals_a = team_a.average_goals_scored() * home_away_factor * performance_factor * competition_factor
    avg_goals_b = team_b.average_goals_scored() * (1 / home_away_factor) * performance_factor * competition_factor  # Fator para o visitante

    # Ajuste baseado no desempenho ofensivo e defensivo dos times
    avg_goals_a_attack = team_a.average_goals_scored() * home_away_factor
    avg_goals_b_defense = team_b.average_goals_conceded() * (1 / home_away_factor)  # Defesa do time visitante

    avg_goals_b_attack = team_b.average_goals_scored() * performance_factor
    avg_goals_a_defense = team_a.average_goals_conceded() * performance_factor  # Defesa do time da casa

    # Aumento das médias se os times têm bom ataque ou defesa
    avg_total_goals = (avg_goals_a + avg_goals_b) / 2
    if avg_goals_a_attack > 1.8:
        avg_total_goals *= 1.05  # Aumenta um pouco para times com bom ataque
    if avg_goals_b_attack > 1.8:
        avg_total_goals *= 1.05
    if avg_goals_a_defense < 1.0:
        avg_total_goals *= 1.05  # Aumenta para equipes com defesa mais fraca
    if avg_goals_b_defense < 1.0:
        avg_total_goals *= 1.05

    # Mercado de Over/Under (por 0.5, 1.5, 2.5 até 8.5)
    over_under_markets = {0.5: {"over": 0, "under": 0},
                          1.5: {"over": 0, "under": 0},
                          2.5: {"over": 0, "under": 0},
                          3.5: {"over": 0, "under": 0},
                          4.5: {"over": 0, "under": 0},
                          5.5: {"over": 0, "under": 0},
                          6.5: {"over": 0, "under": 0},
                          7.5: {"over": 0, "under": 0},
                          8.5: {"over": 0, "under": 0}}

    # Calculando as probabilidades de cada mercado (Over/Under)
    for market in over_under_markets.keys():
        prob_under = poisson.cdf(market, avg_total_goals)
        prob_over = 1 - prob_under

        # Normalizando para garantir que as probabilidades de 'over' e 'under' somem 100%
        total_prob = prob_over + prob_under
        prob_over_percent = (prob_over / total_prob) * 100
        prob_under_percent = (prob_under / total_prob) * 100

        over_under_markets[market] = {"over": prob_over_percent, "under": prob_under_percent}

    # Adicionando um fator de incerteza se a média de gols for muito baixa (baixa amostra)
    if avg_total_goals < 1.5:
        for market in over_under_markets:
            over_under_markets[market]["over"] *= 1.05  # Aumentando o Over se a média de gols for muito baixa
            over_under_markets[market]["under"] *= 0.95  # Reduzindo o Under

    return over_under_markets