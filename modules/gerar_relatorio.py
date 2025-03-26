import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def gerar_relatorio(team_a, team_b, h2h_stats, prob_market, resultados, recent_games_a, recent_games_b, home_away_stats_a, home_away_stats_b):
    """
    Gera um relatório detalhado com análise estatística e tendências, e salva como um arquivo de texto.
    """
    try:
        # Verificações básicas
        if not recent_games_a or not recent_games_b:
            raise ValueError("Dados insuficientes para gerar o relatório. Verifique os jogos recentes dos times.")

        # Definir nome de arquivo com base na data e hora
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        arquivo_relatorio = f"relatorio_jogo_{timestamp}.txt"

        with open(arquivo_relatorio, "w", encoding="utf-8") as f:
            # Cabeçalho do Relatório
            f.write(f"Relatório de Análise: {team_a['name']} vs {team_b['name']}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")

            # Estatísticas gerais de desempenho
            f.write(f"⚽ Desempenho Ofensivo:\n")
            f.write(f"{team_a['name']}: {team_a['average_goals_scored']:.2f} gols/jogo\n")
            f.write(f"{team_b['name']}: {team_b['average_goals_scored']:.2f} gols/jogo\n\n")

            f.write(f"🛡️ Desempenho Defensivo:\n")
            f.write(f"{team_a['name']}: {team_a['average_goals_conceded']:.2f} gols sofridos/jogo\n")
            f.write(f"{team_b['name']}: {team_b['average_goals_conceded']:.2f} gols sofridos/jogo\n\n")

            # Forma Recente dos Times
            f.write(f"📊 Forma Recente (Últimos 5 Jogos):\n")
            f.write(f"{team_a['name']}: {sum(recent_games_a) / max(len(recent_games_a), 1):.2f} gols marcados, "
                    f"{sum(recent_games_b) / max(len(recent_games_b), 1):.2f} gols sofridos\n")
            f.write(f"{team_b['name']}: {sum(recent_games_b) / max(len(recent_games_b), 1):.2f} gols marcados, "
                    f"{sum(recent_games_b) / max(len(recent_games_b), 1):.2f} gols sofridos\n\n")

            # Estatísticas de H2H
            if h2h_stats:
                f.write("⚔️ Estatísticas de Confrontos Diretos (H2H):\n")
                for chave, valor in h2h_stats.items():
                    f.write(f"{chave.capitalize()}: {valor}\n")
            else:
                f.write("Nenhum confronto direto encontrado.\n\n")

            # Resultados simulados
            f.write(f"\n🎲 Simulação de Partida: {resultados['gols_a']} x {resultados['gols_b']}\n")

            # Probabilidades de Mercado
            f.write("\n📈 Probabilidades de Mercado:\n")
            for mercado, valores in prob_market.items():
                f.write(f"Over/Under {mercado}: {valores.get('over', 0) * 100:.2f}% / {valores.get('under', 0) * 100:.2f}%\n")

            # Análise do Impacto do Local (Casa/Fora)
            f.write("\n🏠 Impacto do Local (Casa/Fora):\n")
            f.write(f"{team_a['name']} (Casa): {home_away_stats_a.get('home', 0):.2f} pontos/média por jogo\n")
            f.write(f"{team_b['name']} (Fora): {home_away_stats_b.get('away', 0):.2f} pontos/média por jogo\n")

            # Gerar gráficos e salvar como imagem
            gerar_graficos(team_a, team_b, prob_market, recent_games_a, recent_games_b, h2h_stats, resultados, arquivo_relatorio)

        print(f"\n✅ Relatório salvo como '{arquivo_relatorio}'")
    except Exception as e:
        print(f"❌ Ocorreu um erro ao gerar o relatório: {e}")


def gerar_graficos(team_a, team_b, prob_market, recent_games_a, recent_games_b, h2h_stats, resultados, arquivo_relatorio):
    """
    Gera gráficos de comparação entre os times e salva como imagens.
    """
    try:
        # Gráfico de Probabilidades Over/Under
        plt.figure(figsize=(10, 6))
        sns.barplot(x=["Over", "Under"], y=[prob_market.get('over', 0), prob_market.get('under', 0)], palette="viridis")
        plt.title(f"Probabilidade de Mercado: Over/Under para {team_a['name']} vs {team_b['name']}")
        plt.ylabel("Probabilidade (%)")
        plt.savefig(f"grafico_probabilidade_{team_a['name']}_{team_b['name']}.png")
        plt.close()

        # Gráfico de Desempenho Ofensivo/Defensivo
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([team_a['name'], team_b['name']], [team_a['average_goals_scored'], team_b['average_goals_scored']], color='green', label='Ofensivo')
        ax.barh([team_a['name'], team_b['name']], [team_a['average_goals_conceded'], team_b['average_goals_conceded']], color='red', label='Defensivo')
        plt.title("Desempenho Ofensivo e Defensivo")
        plt.xlabel("Média de Gols")
        plt.legend()
        plt.savefig(f"grafico_desempenho_{team_a['name']}_{team_b['name']}.png")
        plt.close()

        # Gráfico de H2H
        if h2h_stats:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar([team_a['name'], team_b['name']], [h2h_stats.get('vitorias_a', 0), h2h_stats.get('vitorias_b', 0)], color='blue')
            plt.title(f"Histórico de Confrontos Diretos (H2H): {team_a['name']} vs {team_b['name']}")
            plt.ylabel("Número de Vitórias")
            plt.savefig(f"grafico_h2h_{team_a['name']}_{team_b['name']}.png")
            plt.close()

        # Gráfico de Gols Marcados Recentemente
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(recent_games_a, label=f'{team_a["name"]} - Últimos Jogos', marker='o')
        ax.plot(recent_games_b, label=f'{team_b["name"]} - Últimos Jogos', marker='o')
        plt.title("Desempenho Recente - Últimos 5 Jogos")
        plt.xlabel("Jogo")
        plt.ylabel("Gols Marcados")
        plt.legend()
        plt.savefig(f"grafico_desempenho_recente_{team_a['name']}_{team_b['name']}.png")
        plt.close()

        # Adicionar os gráficos ao relatório como anexos
        with open(arquivo_relatorio, "a", encoding="utf-8") as f:
            f.write("\n📊 Gráficos Anexados:\n")
            f.write("Gráfico de Probabilidades: grafico_probabilidade.png\n")
            f.write("Gráfico de Desempenho: grafico_desempenho.png\n")
            f.write("Gráfico H2H: grafico_h2h.png\n")
            f.write("Gráfico Desempenho Recente: grafico_desempenho_recente.png\n")

    except Exception as e:
        print(f"❌ Erro ao gerar gráficos: {e}")