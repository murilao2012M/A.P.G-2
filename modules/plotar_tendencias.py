from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plotar_tendencias(team_a, team_b, n_ultimos_jogos=10, suavizar=False, window_size=3, plot_acumulado=False, 
                      marcar_eventos=False, salvar_imagem=False, interactive=False):
    """
    Plota tendências de desempenho ofensivo e defensivo dos dois times, com várias opções de personalização.
    
    Argumentos:
    - n_ultimos_jogos: Número de jogos mais recentes a serem considerados.
    - suavizar: Se True, aplica uma média móvel para suavizar as tendências.
    - window_size: Tamanho da janela para suavização (média móvel), se 'suavizar' for True.
    - plot_acumulado: Se True, plota o desempenho acumulado ao longo do tempo.
    - marcar_eventos: Se True, marca vitórias/derrotas significativas nos gráficos.
    - salvar_imagem: Se True, salva o gráfico como uma imagem (PNG, SVG, etc.).
    - interactive: Se True, habilita a interatividade (somente se estiver usando `mplcursors`).
    """
    # Validando a quantidade de jogos a ser exibida
    n_ultimos_jogos = min(n_ultimos_jogos, len(team_a.golsMarcados), len(team_b.golsMarcados))

    jogos = range(1, n_ultimos_jogos + 1)
    
    # Calculando médias móveis, se desejado
    if suavizar:
        window = min(window_size, n_ultimos_jogos)
        team_a.golsMarcados = np.convolve(team_a.golsMarcados[-n_ultimos_jogos:], np.ones(window)/window, mode='valid')
        team_a.golsSofridos = np.convolve(team_a.golsSofridos[-n_ultimos_jogos:], np.ones(window)/window, mode='valid')
        team_b.golsMarcados = np.convolve(team_b.golsMarcados[-n_ultimos_jogos:], np.ones(window)/window, mode='valid')
        team_b.golsSofridos = np.convolve(team_b.golsSofridos[-n_ultimos_jogos:], np.ones(window)/window, mode='valid')
    
    # Calculando desempenho acumulado, se desejado
    if plot_acumulado:
        team_a.golsMarcados = np.cumsum(team_a.golsMarcados[-n_ultimos_jogos:])
        team_a.golsSofridos = np.cumsum(team_a.golsSofridos[-n_ultimos_jogos:])
        team_b.golsMarcados = np.cumsum(team_b.golsMarcados[-n_ultimos_jogos:])
        team_b.golsSofridos = np.cumsum(team_b.golsSofridos[-n_ultimos_jogos:])

    # Iniciando o gráfico
    plt.figure(figsize=(12, 6))

    # Plotando as tendências para o Time A
    plt.plot(jogos, team_a.golsMarcados, label=f"{team_a.name} - Gols Marcados", marker='o', color='blue')
    plt.plot(jogos, team_a.golsSofridos, label=f"{team_a.name} - Gols Sofridos", marker='o', linestyle='--', color='blue')

    # Plotando as tendências para o Time B
    plt.plot(jogos, team_b.golsMarcados, label=f"{team_b.name} - Gols Marcados", marker='o', color='red')
    plt.plot(jogos, team_b.golsSofridos, label=f"{team_b.name} - Gols Sofridos", marker='o', linestyle='--', color='red')

    # Marcar eventos importantes (vitórias/derrotas)
    if marcar_eventos:
        eventos_a = ["Vitória" if team_a.golsMarcados[i] > team_a.golsSofridos[i] else "Derrota" for i in range(n_ultimos_jogos)]
        eventos_b = ["Vitória" if team_b.golsMarcados[i] > team_b.golsSofridos[i] else "Derrota" for i in range(n_ultimos_jogos)]

        for i in range(n_ultimos_jogos):
            if eventos_a[i] == "Vitória":
                plt.plot(jogos[i], team_a.golsMarcados[i], 'go', markersize=8)  # Marcar vitória com ponto verde
            if eventos_a[i] == "Derrota":
                plt.plot(jogos[i], team_a.golsMarcados[i], 'ro', markersize=8)  # Marcar derrota com ponto vermelho
            if eventos_b[i] == "Vitória":
                plt.plot(jogos[i], team_b.golsMarcados[i], 'go', markersize=8)
            if eventos_b[i] == "Derrota":
                plt.plot(jogos[i], team_b.golsMarcados[i], 'ro', markersize=8)

    # Adicionando título e rótulos
    plt.title(f"Tendências de Desempenho: {team_a.name} vs {team_b.name}", fontsize=16)
    plt.xlabel(f"Últimos {n_ultimos_jogos} Jogos", fontsize=12)
    plt.ylabel("Gols", fontsize=12)
    
    # Exibindo a legenda
    plt.legend()

    # Adicionando a grade
    plt.grid(True)

    # Ajustando o layout para não cortar os rótulos
    plt.tight_layout()

    # # Tornando interativo com mplcursors (apenas se for True e se a biblioteca estiver instalada)
    # if interactive:
    #     import mplcursors
    #     mplcursors.cursor(hover=True)

    # Exibindo o gráfico
    plt.show()

    # Salvando a imagem se solicitado
    if salvar_imagem:
        plt.savefig(f"{team_a.name}_vs_{team_b.name}_tendencias.png", dpi=300)

### Função de Análises Comparativas Avançadas
def analises_comparativas(team_a, team_b, n_ultimos_jogos=10):
    """
    Realiza análises comparativas detalhadas, como a diferença de gols e porcentagem de vitórias/empates/derrotas.
    """
    # Comparação de gols
    gols_marcados_a = sum(team_a.golsMarcados[-n_ultimos_jogos:])
    gols_sofridos_a = sum(team_a.golsSofridos[-n_ultimos_jogos:])
    gols_marcados_b = sum(team_b.golsMarcados[-n_ultimos_jogos:])
    gols_sofridos_b = sum(team_b.golsSofridos[-n_ultimos_jogos:])
    
    print(f"\nAnálise Comparativa entre {team_a.name} e {team_b.name}:")
    
    # Diferença de gols
    diff_gols_a = gols_marcados_a - gols_sofridos_a
    diff_gols_b = gols_marcados_b - gols_sofridos_b
    print(f"{team_a.name} - Diferença de Gols: {diff_gols_a}")
    print(f"{team_b.name} - Diferença de Gols: {diff_gols_b}")
    
    # Calculando vitórias/empates/derrotas
    vitoria_a = sum([1 for i in range(n_ultimos_jogos) if team_a.golsMarcados[i] > team_a.golsSofridos[i]])
    derrota_a = sum([1 for i in range(n_ultimos_jogos) if team_a.golsMarcados[i] < team_a.golsSofridos[i]])
    empate_a = n_ultimos_jogos - vitoria_a - derrota_a

    vitoria_b = sum([1 for i in range(n_ultimos_jogos) if team_b.golsMarcados[i] > team_b.golsSofridos[i]])
    derrota_b = sum([1 for i in range(n_ultimos_jogos) if team_b.golsMarcados[i] < team_b.golsSofridos[i]])
    empate_b = n_ultimos_jogos - vitoria_b - derrota_b

    print(f"\n{team_a.name} - Vitória: {vitoria_a} / Empate: {empate_a} / Derrota: {derrota_a}")
      # Cálculo para o Time A
    vitoria_a = sum([1 for i in range(n_ultimos_jogos) if team_a.golsMarcados[i] > team_a.golsSofridos[i]])
    derrota_a = sum([1 for i in range(n_ultimos_jogos) if team_a.golsMarcados[i] < team_a.golsSofridos[i]])
    empate_a = n_ultimos_jogos - vitoria_a - derrota_a

    # Cálculo para o Time B
    vitoria_b = sum([1 for i in range(n_ultimos_jogos) if team_b.golsMarcados[i] > team_b.golsSofridos[i]])
    derrota_b = sum([1 for i in range(n_ultimos_jogos) if team_b.golsMarcados[i] < team_b.golsSofridos[i]])
    empate_b = n_ultimos_jogos - vitoria_b - derrota_b

    # Exibindo as informações de vitórias, empates e derrotas de cada time
    print(f"\nAnálise Comparativa entre {team_a.name} e {team_b.name}:")
    print(f"{team_a.name} - Vitória: {vitoria_a} / Empate: {empate_a} / Derrota: {derrota_a}")
    print(f"{team_b.name} - Vitória: {vitoria_b} / Empate: {empate_b} / Derrota: {derrota_b}")

    # Exibindo a diferença de gols entre os dois times
    diff_gols_a = gols_marcados_a - gols_sofridos_a
    diff_gols_b = gols_marcados_b - gols_sofridos_b
    print(f"\n{team_a.name} - Diferença de Gols: {diff_gols_a}")
    print(f"{team_b.name} - Diferença de Gols: {diff_gols_b}")

    # Calculando a porcentagem de vitórias para cada time
    pct_vitoria_a = (vitoria_a / n_ultimos_jogos) * 100
    pct_empate_a = (empate_a / n_ultimos_jogos) * 100
    pct_derrota_a = (derrota_a / n_ultimos_jogos) * 100

    pct_vitoria_b = (vitoria_b / n_ultimos_jogos) * 100
    pct_empate_b = (empate_b / n_ultimos_jogos) * 100
    pct_derrota_b = (derrota_b / n_ultimos_jogos) * 100

    # Exibindo a porcentagem de vitórias, empates e derrotas
    print(f"\n{team_a.name} - Porcentagem de Vitória: {pct_vitoria_a:.2f}% / Empate: {pct_empate_a:.2f}% / Derrota: {pct_derrota_a:.2f}%")
    print(f"{team_b.name} - Porcentagem de Vitória: {pct_vitoria_b:.2f}% / Empate: {pct_empate_b:.2f}% / Derrota: {pct_derrota_b:.2f}%")