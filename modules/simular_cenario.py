import random
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

def simular_cenario(team_a, team_b, n_simulacoes=1000, seed=42):
    """
    Simula uma partida entre duas equipes ajustando os fatores ofensivos e defensivos de maneira realista,
    com a opção de realizar várias simulações para obter uma distribuição de resultados.
    Agora com seed para garantir resultados determinísticos e uma análise mais completa com várias simulações.
    """
    print("\n🎮 Simulação de Cenário da Partida entre Time Mandante e Visitante:")

    # Definir a semente para garantir resultados determinísticos
    if seed is not None:
        random.seed(seed)  # Garantindo que o random seja determinístico
        np.random.seed(seed)  # Garantindo que a numpy também seja determinística

    # Ajuste de fatores ofensivos e defensivos (melhorando com base no histórico recente)
    def obter_fator(nome_time, tipo):
        if tipo == "ataque":
            return 1.0 + (0.2 * team_a.average_goals_scored() if nome_time == team_a.name else 0.2 * team_b.average_goals_scored())
        elif tipo == "defesa":
            return 1.0 - (0.1 * team_a.average_goals_conceded() if nome_time == team_a.name else 0.1 * team_b.average_goals_conceded())

    ataque_a = obter_fator(team_a.name, "ataque")
    ataque_b = obter_fator(team_b.name, "ataque")
    defesa_a = obter_fator(team_a.name, "defesa")
    defesa_b = obter_fator(team_b.name, "defesa")

    # Calcular médias ajustadas de gols
    media_a = team_a.average_goals_scored() * ataque_a
    media_b = team_b.average_goals_scored() * ataque_b

    impacto_defesa_a = max(0.8, min(defesa_a, 1.2))  # Limitar a defesa entre 0.8 e 1.2
    impacto_defesa_b = max(0.8, min(defesa_b, 1.2))  # Limitar a defesa entre 0.8 e 1.2

    avg_goals_a = max(0.5, min(media_a * impacto_defesa_b, 4))  # Limitar a média ajustada entre 0.5 e 4
    avg_goals_b = max(0.5, min(media_b * impacto_defesa_a, 4))  # Limitar a média ajustada entre 0.5 e 4

    print(f"\n⚙️ Ajustes: {team_a.name} com ataque {ataque_a:.2f}, defesa {defesa_a:.2f}, e {team_b.name} com ataque {ataque_b:.2f}, defesa {defesa_b:.2f}.")
    print(f"⚽ Média ajustada de gols: {team_a.name} {avg_goals_a:.2f} vs {team_b.name} {avg_goals_b:.2f}.")

    # Simulação de múltiplos cenários
    resultados_a = []
    resultados_b = []

    for _ in range(n_simulacoes):
        gols_a = poisson.rvs(avg_goals_a)
        gols_b = poisson.rvs(avg_goals_b)
        resultados_a.append(gols_a)
        resultados_b.append(gols_b)

        print(f"⚽ Simulação {_ + 1}: {team_a.name} {gols_a} x {gols_b} {team_b.name}")

    # Análise dos resultados
    vitorias_a = sum([1 for i in range(n_simulacoes) if resultados_a[i] > resultados_b[i]])
    empates = sum([1 for i in range(n_simulacoes) if resultados_a[i] == resultados_b[i]])
    vitorias_b = sum([1 for i in range(n_simulacoes) if resultados_b[i] > resultados_a[i]])

    print(f"\n🏆 Resultados após {n_simulacoes} simulações:")
    print(f"Vitórias {team_a.name}: {vitorias_a} ({(vitorias_a/n_simulacoes)*100:.2f}%)")
    print(f"Empates: {empates} ({(empates/n_simulacoes)*100:.2f}%)")
    print(f"Vitórias {team_b.name}: {vitorias_b} ({(vitorias_b/n_simulacoes)*100:.2f}%)")

    # Exibindo a distribuição de gols
    plt.figure(figsize=(10, 6))
    plt.hist(resultados_a, bins=range(int(min(resultados_a)), int(max(resultados_a)) + 2), alpha=0.7, label=f'{team_a.name} - Gols')
    plt.hist(resultados_b, bins=range(int(min(resultados_b)), int(max(resultados_b)) + 2), alpha=0.7, label=f'{team_b.name} - Gols')
    plt.xlabel('Número de Gols')
    plt.ylabel('Frequência')
    plt.title(f'Distribuição de Gols Simulados ({team_a.name} vs {team_b.name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Resultado final da simulação
    return vitorias_a, empates, vitorias_b