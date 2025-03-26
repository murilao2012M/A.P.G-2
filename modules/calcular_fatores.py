from modules.calculate_recent_performance import calculate_recent_performance

def calcular_fatores(team_a, team_b, dados_partidas, peso_recente=1.0, peso_adversario=0.5, peso_moral=0.2, peso_mandante=0.1):
    """
    Calculates the relative performance factors for both teams.
    Now includes form trends, opponent strength, morale factor, and home advantage dynamically.

    Parameters:
        team_a (TeamStats): Object of team A.
        team_b (TeamStats): Object of team B.
        dados_partidas (dict): Historical match data.
        peso_recente (float): Weight assigned to recent performance.
        peso_adversario (float): Weight assigned to opponent performance.
        peso_moral (float): Weight assigned to team morale.
        peso_mandante (float): Weight assigned to home advantage.

    Returns:
        tuple: Adjusted performance values for both teams.
    """

    # Recent Performance Analysis
    recent_a_goals, recent_a_conceded = calculate_recent_performance(dados_partidas, team_a, home_games=True)
    recent_b_goals, recent_b_conceded = calculate_recent_performance(dados_partidas, team_b, home_games=False)

    # Opponent Strength Consideration
    opponent_a_goals, opponent_a_conceded = calculate_recent_performance(dados_partidas, team_b, home_games=False)
    opponent_b_goals, opponent_b_conceded = calculate_recent_performance(dados_partidas, team_a, home_games=True)

    # Adjusting performance dynamically
    performance_a = team_a.average_goals_scored() - team_b.average_goals_conceded() + (recent_a_goals * peso_recente)
    performance_b = team_b.average_goals_scored() - team_a.average_goals_conceded() + (recent_b_goals * peso_recente)

    # Strength of Opponent Adjustment
    opponent_factor_a = (opponent_a_goals - opponent_b_conceded) * peso_adversario
    opponent_factor_b = (opponent_b_goals - opponent_a_conceded) * peso_adversario

    performance_a += opponent_factor_a
    performance_b += opponent_factor_b

    # Morale Factor (Winning Boost, Losing Penalty)
    morale_a = (team_a.recent_wins() * 0.1) - (team_a.recent_losses() * 0.1)
    morale_b = (team_b.recent_wins() * 0.1) - (team_b.recent_losses() * 0.1)

    performance_a += morale_a * peso_moral
    performance_b += morale_b * peso_moral

    # Home Advantage Factor
    if team_a.is_playing_home():
        performance_a += peso_mandante  # Boost for home games
        performance_b -= peso_mandante  # Penalty for away games
    elif team_b.is_playing_home():
        performance_b += peso_mandante
        performance_a -= peso_mandante

    return performance_a, performance_b


def ajustar_fatores(team, tipo="ataque", intensidade=True, impacto_lesao=False, intensidade_lesao=0.05, considerar_localizacao=False):
    """
    Dynamically adjusts attack or defense factors based on:
      - Recent results
      - Win/loss intensity
      - Injury impact
      - Home/Away performance differences

    Parameters:
        team (TeamStats): Object of the team.
        tipo (str): "ataque" (attack) or "defesa" (defense).
        intensidade (bool): If True, considers intensity of recent results.
        impacto_lesao (bool): If True, considers injuries in calculations.
        intensidade_lesao (float): Intensity of injury impact.
        considerar_localizacao (bool): If True, adjusts for home/away advantage.

    Returns:
        float: Adjusted factor for attack or defense.
    """
    
    fator = 1.0  # Base value
    
    if tipo == "ataque":
        # Boost attack factor based on recent wins
        fator += team.recent_wins() * 0.05
        fator -= team.recent_draws() * 0.02  # Small penalty for draws
        
        # Increase factor based on recent offensive performance
        if intensidade:
            fator += sum(team.golsMarcados[-5:]) * 0.01

        # Injury impact on attacking power
        if impacto_lesao and team.has_injuries():
            fator -= sum(team.golsMarcados[-5:]) * intensidade_lesao

    elif tipo == "defesa":
        # Reduce defensive factor for recent losses
        fator -= team.recent_losses() * 0.05

        # Increase factor based on defensive performance
        if intensidade:
            fator -= sum(team.golsSofridos[-5:]) * 0.01

        # Injury impact on defensive power
        if impacto_lesao and team.has_injuries():
            fator -= sum(team.golsSofridos[-5:]) * intensidade_lesao

    # Adjusting for Home/Away Performance
    if considerar_localizacao:
        if team.is_playing_home():
            fator *= 1.1  # Home boost
        else:
            fator *= 0.9  # Away penalty

    return max(0.5, min(fator, 1.5))  # Ensuring the factor stays within a reasonable range