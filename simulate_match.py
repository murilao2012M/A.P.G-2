import numpy as np
from scipy.stats import poisson

def simulate_match(team_a, team_b, home_advantage=0.1, recent_performance_weight=1.5, 
                   confidence_level=0.95, n_simulations=500, seed=None):
    """
    Simula uma partida entre dois times ajustando pela força ofensiva e defensiva,
    vantagem de jogar em casa, desempenho recente das equipes e aplica ajustes para limitar aleatoriedade.
    Utiliza uma abordagem baseada em distribuições probabilísticas para gerar resultados mais realistas.
    
    Arguments:
    - team_a: objeto que contém as estatísticas do time A.
    - team_b: objeto que contém as estatísticas do time B.
    - home_advantage: vantagem de jogar em casa (como ajuste na média de gols).
    - recent_performance_weight: peso para o desempenho recente das equipes.
    - confidence_level: nível de certeza que influencia a variabilidade (0-1).
    - n_simulations: número de simulações para gerar uma distribuição de resultados.
    - seed: semente para reprodutibilidade.
    
    Returns:
    - resultado final simulado (gols_a, gols_b)
    """
    # Definir a semente se necessário para resultados reprodutíveis
    if seed is not None:
        np.random.seed(seed)

    # Ajustes pela média de gols e desempenho recente
    avg_goals_a = team_a.average_goals_scored() * recent_performance_weight
    avg_goals_b = team_b.average_goals_scored() * recent_performance_weight

    # Ajuste de vantagem de jogar em casa
    avg_goals_a += home_advantage

    # Ajustes defensivos baseados nos gols concedidos
    defense_factor_a = max(0.5, 1 - team_a.average_goals_conceded())
    defense_factor_b = max(0.5, 1 - team_b.average_goals_conceded())

    # Ajustando médias ofensivas com fator defensivo do adversário
    adjusted_goals_a = avg_goals_a * defense_factor_b
    adjusted_goals_b = avg_goals_b * defense_factor_a

    # Limitações de gols para evitar extremos, mas usando uma distribuição mais flexível
    adjusted_goals_a = max(0.5, min(adjusted_goals_a, 4))
    adjusted_goals_b = max(0.5, min(adjusted_goals_b, 4))

    # Simulação de múltiplos cenários utilizando uma distribuição de Poisson
    sim_results_a = poisson.rvs(adjusted_goals_a, size=n_simulations)
    sim_results_b = poisson.rvs(adjusted_goals_b, size=n_simulations)

    # Calcular as probabilidades de vitória, empate e derrota com base nas simulações
    vitoria_a = np.sum(sim_results_a > sim_results_b)
    empate = np.sum(sim_results_a == sim_results_b)
    vitoria_b = np.sum(sim_results_b > sim_results_a)

    prob_vitoria_a = vitoria_a / n_simulations
    prob_empate = empate / n_simulations
    prob_vitoria_b = vitoria_b / n_simulations

    print(f"\n🏆 Resultados após {n_simulations} simulações:")
    print(f"Vitórias {team_a.name}: {vitoria_a} ({prob_vitoria_a*100:.2f}%)")
    print(f"Empates: {empate} ({prob_empate*100:.2f}%)")
    print(f"Vitórias {team_b.name}: {vitoria_b} ({prob_vitoria_b*100:.2f}%)")

    # Retornar os resultados finais (média de gols por time nas simulações)
    gols_a_final = np.mean(sim_results_a)
    gols_b_final = np.mean(sim_results_b)

    # Exibir a média de gols
    print(f"\n⚽ Média de gols simulados: {team_a.name} {gols_a_final:.2f} x {gols_b_final:.2f} {team_b.name}")

    return gols_a_final, gols_b_final, prob_vitoria_a, prob_empate, prob_vitoria_b

# Exemplo de uso da função:
# Supondo que você tenha objetos `team_a` e `team_b` com as funções `average_goals_scored()` e `average_goals_conceded()` já implementadas.