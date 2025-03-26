#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Simulação de Futebol – Versão Refatorada para Console
Melhorias:
  • Removeu o PySimpleGUI e passou a utilizar entrada/saída via console
  • Os menus são apresentados no console, com opções numéricas
  • Novas funcionalidades: impacto do tempo, impacto da fadiga, simulação de torneio, etc.
  • O código original foi mantido integralmente, apenas adaptado para console
"""
sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))
import math
import random
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom, skellam
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sys
from io import StringIO

# Importa funções dos módulos (assegure-se de que esses módulos estejam na pasta "modules")
from modules.calcular_fatores import ajustar_fatores
from modules.calcular_recorrencia_gols import calcular_recorrencia_gols
from modules.calculate_goal_market_probabilities import calculate_goal_market_probabilities
from modules.calculate_h2h_statistics import calculate_h2h_statistics
from modules.carregar_dados_excel import carregar_dados_excel
from modules.comparar_times import comparar_times
from modules.exibir_xg import exibir_xg
from modules.plotar_probabilidades_mercado import plotar_probabilidades_mercado
from modules.plotar_tendencias import plotar_tendencias
from modules.simular_cenario import simular_cenario
from modules.simulate_match import simulate_match

# =============================================================================
# FUNÇÕES ORIGINAIS (adaptadas para console)
# =============================================================================

def save_to_excel(data, filename="registros_jogos.xlsx"):
    try:
        existing_data = pd.read_excel(filename)
        new_data = pd.concat([existing_data, data], ignore_index=True).drop_duplicates()
    except FileNotFoundError:
        new_data = data
    new_data.to_excel(filename, index=False)
    print("✅ Dados atualizados em", filename)

def mostrar_escudo(time):
    caminho_escudo = f'escudos/{time}.png'
    if os.path.exists(caminho_escudo):
        img = Image.open(caminho_escudo)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Escudo do {time}", fontsize=14, weight='bold')
        plt.show()
    else:
        print(f"⚠️ Escudo do {time} não encontrado.")

class TeamStats:
    def __init__(self, name, golsMarcados, golsSofridos):
        if not isinstance(golsMarcados, list) or not isinstance(golsSofridos, list):
            raise ValueError("golsMarcados e golsSofridos devem ser listas.")
        self.name = name
        self.golsMarcados = golsMarcados
        self.golsSofridos = golsSofridos

    def average_goals_scored(self):
        if not self.golsMarcados:
            print(f"⚠️ Não há dados de gols marcados para {self.name}.")
            return 0
        return sum(self.golsMarcados) / len(self.golsMarcados)

    def average_goals_conceded(self):
        if not self.golsSofridos:
            print(f"⚠️ Não há dados de gols sofridos para {self.name}.")
            return 0
        return sum(self.golsSofridos) / len(self.golsSofridos)

    def last_n_game_performance(self, n=5):
        n = min(n, len(self.golsMarcados), len(self.golsSofridos))
        if n == 0:
            print(f"⚠️ Dados insuficientes para {self.name}.")
            return 0, 0
        return sum(self.golsMarcados[-n:]) / n, sum(self.golsSofridos[-n:]) / n

    def recent_wins(self):
        return sum(1 for gm, gs in zip(self.golsMarcados[-10:], self.golsSofridos[-10:]) if gm > gs)

    def recent_draws(self):
        return sum(1 for gm, gs in zip(self.golsMarcados[-10:], self.golsSofridos[-10:]) if gm == gs)

    def recent_losses(self):
        return sum(1 for gm, gs in zip(self.golsMarcados[-10:], self.golsSofridos[-10:]) if gm < gs)

    def eficiencia_ofensiva(self):
        return self.average_goals_scored()

    def eficiencia_defensiva(self):
        return self.average_goals_conceded()

    def prever_gols(self, time_adversario):
        if not self.golsMarcados or not time_adversario.golsSofridos:
            print("⚠️ Dados insuficientes para previsão de gols.")
            return 0
        mean_gols = (self.average_goals_scored() + time_adversario.average_goals_conceded()) / 2
        return np.random.poisson(mean_gols)

    def simular_partida_monte_carlo(self, time_adversario, n_simulacoes=5000):
        try:
            resultados = [simulate_match(self, time_adversario) for _ in range(n_simulacoes)]
        except Exception as e:
            print(f"Erro na simulação de partida: {e}")
            return 0, 0, 0
        vitorias = sum(1 for g_a, g_b in resultados if g_a > g_b)
        empates = sum(1 for g_a, g_b in resultados if g_a == g_b)
        derrotas = sum(1 for g_a, g_b in resultados if g_a < g_b)
        prob_vitoria = (vitorias / n_simulacoes) * 100
        prob_empate = (empates / n_simulacoes) * 100
        prob_derrota = (derrotas / n_simulacoes) * 100
        print(f"Probabilidades ({n_simulacoes} simulações):\n"
              f"🔹 {self.name} Vitória: {prob_vitoria:.2f}%\n"
              f"🔸 Empate: {prob_empate:.2f}%\n"
              f"🔻 {time_adversario.name} Vitória: {prob_derrota:.2f}%")
        plt.figure(figsize=(6, 4))
        plt.bar(['Vitória', 'Empate', 'Derrota'], [prob_vitoria, prob_empate, prob_derrota])
        plt.ylabel('Probabilidade (%)')
        plt.title(f"Simulação Monte Carlo: {self.name} vs {time_adversario.name}")
        plt.show()
        return prob_vitoria, prob_empate, prob_derrota

    def adicionar_resultado(self, gols_marcados, gols_sofridos):
        if not isinstance(gols_marcados, (int, float)) or not isinstance(gols_sofridos, (int, float)):
            raise ValueError("Os valores de gols devem ser numéricos.")
        self.golsMarcados.append(gols_marcados)
        self.golsSofridos.append(gols_sofridos)
        print(f"✅ Resultado adicionado para {self.name}: {gols_marcados} marcados, {gols_sofridos} sofridos.")

    def average_goals_scored_weighted(self, weight_factor=0.9):
        if not self.golsMarcados:
            print(f"⚠️ Não há dados para {self.name}.")
            return 0
        pesos = [weight_factor ** i for i in range(len(self.golsMarcados))]
        pesos.reverse()
        total_pesos = sum(pesos)
        return sum(g * p for g, p in zip(self.golsMarcados, pesos)) / total_pesos

    def average_goals_conceded_weighted(self, weight_factor=0.9):
        if not self.golsSofridos:
            print(f"⚠️ Não há dados para {self.name}.")
            return 0
        pesos = [weight_factor ** i for i in range(len(self.golsSofridos))]
        pesos.reverse()
        total_pesos = sum(pesos)
        return sum(g * p for g, p in zip(self.golsSofridos, pesos)) / total_pesos

# =============================================================================
# OUTRAS FUNÇÕES ORIGINAIS
# =============================================================================

def compute_basic_expected_values(team_a, team_b, home_advantage=0.1, recent_performance_weight=1.5, confidence_level=0.95):
    avg_goals_a = team_a.average_goals_scored() * recent_performance_weight + home_advantage
    avg_goals_b = team_b.average_goals_scored() * recent_performance_weight
    defense_factor_a = max(0.5, 1 - team_a.average_goals_conceded())
    defense_factor_b = max(0.5, 1 - team_b.average_goals_conceded())
    expected_a = max(0.5, min(avg_goals_a * defense_factor_b, 5)) * confidence_level
    expected_b = max(0.5, min(avg_goals_b * defense_factor_a, 5)) * confidence_level
    return expected_a, expected_b

def compute_score_probability(expected_a, expected_b, gols_a, gols_b):
    p_a = poisson.pmf(gols_a, expected_a)
    p_b = poisson.pmf(gols_b, expected_b)
    return p_a * p_b

def preparar_dados_para_treinamento(df_completo):
    linhas_treinamento = []
    for _, row in df_completo.iterrows():
        gols_marc_mandante = row['FTHG']
        gols_sofr_mandante = row['FTAG']
        ftr = row['FTR']
        if ftr == 'H':
            vit_mandante, emp_mandante, der_mandante = 1, 0, 0
        elif ftr == 'D':
            vit_mandante, emp_mandante, der_mandante = 0, 1, 0
        else:
            vit_mandante, emp_mandante, der_mandante = 0, 0, 1
        linhas_treinamento.append({
            'Gols Marcados': gols_marc_mandante,
            'Gols Sofridos': gols_sofr_mandante,
            'Vitórias': vit_mandante,
            'Empates': emp_mandante,
            'Derrotas': der_mandante,
            'Resultado': 2 if ftr == 'H' else (1 if ftr == 'D' else 0)
        })
        gols_marc_visitante = row['FTAG']
        gols_sofr_visitante = row['FTHG']
        if ftr == 'A':
            vit_visit, emp_visit, der_visit = 1, 0, 0
        elif ftr == 'D':
            vit_visit, emp_visit, der_visit = 0, 1, 0
        else:
            vit_visit, emp_visit, der_visit = 0, 0, 1
        linhas_treinamento.append({
            'Gols Marcados': gols_marc_visitante,
            'Gols Sofridos': gols_sofr_visitante,
            'Vitórias': vit_visit,
            'Empates': emp_visit,
            'Derrotas': der_visit,
            'Resultado': 2 if ftr == 'A' else (1 if ftr == 'D' else 0)
        })
    return pd.DataFrame(linhas_treinamento)

def train_model_random_forest(df):
    X = df[['Gols Marcados', 'Gols Sofridos', 'Vitórias', 'Empates', 'Derrotas']]
    y = df['Resultado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("=== RandomForest ===")
    print(f"Acurácia (hold-out): {accuracy_score(y_test, y_pred):.2f}")
    print("Relatório de classificação:")
    print(classification_report(y_test, y_pred))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print("Acurácia Cross-Val (5 folds):", f"{round(scores.mean(), 3)} +/- {round(scores.std(), 3)}")
    return model

def train_model_xgboost(df):
    X = df[['Gols Marcados', 'Gols Sofridos', 'Vitórias', 'Empates', 'Derrotas']]
    y = df['Resultado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    print("=== XGBoost ===")
    print(f"Acurácia (hold-out): {accuracy_score(y_test, y_pred):.2f}")
    print("Relatório de classificação:")
    print(classification_report(y_test, y_pred))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')
    print("Acurácia Cross-Val (5 folds):", f"{round(scores.mean(), 3)} +/- {round(scores.std(), 3)}")
    return xgb_model

# =============================================================================
# REGISTRO E EXPORTAÇÃO DE SIMULAÇÕES (console)
# =============================================================================

simulation_records = []

def registrar_resultado_simulacao():
    time_a = input("Informe o nome do Time A: ")
    time_b = input("Informe o nome do Time B: ")
    placar = input("Informe o placar (ex.: 2-1): ")
    mercado = input("Informe o mercado escolhido: ")
    ganhadora = input("Essa simulação foi ganhadora? (s/n): ").lower()
    registro = {
        "Time A": time_a,
        "Time B": time_b,
        "Placar": placar,
        "Mercado": mercado,
        "Ganhadora": "Sim" if ganhadora in ["s", "sim"] else "Não"
    }
    simulation_records.append(registro)
    print("✅ Resultado registrado com sucesso!")

def exibir_registros_simulacoes():
    if not simulation_records:
        print("⚠️ Nenhum registro de simulação encontrado.")
        return
    registros = "\n".join([f"{i+1}. Time A: {reg['Time A']} | Time B: {reg['Time B']} | Placar: {reg['Placar']} | Mercado: {reg['Mercado']} | Ganhadora: {reg['Ganhadora']}"
                           for i, reg in enumerate(simulation_records)])
    print("Registros de Simulações:")
    print(registros)

def export_simulation_report(data, filename="relatorio_simulacoes.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print("✅ Relatório exportado para", filename)

# =============================================================================
# FUNÇÕES DE DISTRIBUIÇÃO, PREVISÕES & ATUALIZAÇÃO DE ELO
# =============================================================================

def poisson_distribution_probabilities(mean, max_goals=5):
    probabilities = {}
    for k in range(0, max_goals + 1):
        probabilities[k] = poisson.pmf(k, mean)
    return probabilities

def skellam_distribution_probability(lambda_a, lambda_b, k):
    return skellam.pmf(k, mu1=lambda_a, mu2=lambda_b)

def predict_outcome_regression(features):
    return {"win": 0.4, "draw": 0.3, "loss": 0.3}

def update_elo_rating(current_rating, opponent_rating, result, k_factor=20):
    expected = 1 / (1 + 10 ** ((opponent_rating - current_rating) / 400))
    new_rating = current_rating + k_factor * (result - expected)
    return new_rating

# =============================================================================
# FUNÇÃO DE ENTRADA DOS DADOS DOS TIMES (Console)
# =============================================================================

def get_teams_console():
    print("== Entrada de Dados dos Times ==")
    team_a_name = input("Nome do Time A: ").strip()
    team_b_name = input("Nome do Time B: ").strip()
    if not team_a_name or not team_b_name:
        print("Erro: Preencha os nomes dos times.")
        return None, None
    team_a_games = []
    team_b_games = []
    print("\nDigite os resultados para o Time A (digite 'fim' para encerrar):")
    while True:
        entrada = input("Gols Marcados e Gols Sofridos (ex.: 2 1): ").strip()
        if entrada.lower() == "fim":
            break
        try:
            gm, gs = map(int, entrada.split())
            team_a_games.append((gm, gs))
        except Exception as e:
            print("Entrada inválida. Tente novamente.", e)
    print("\nDigite os resultados para o Time B (digite 'fim' para encerrar):")
    while True:
        entrada = input("Gols Marcados e Gols Sofridos (ex.: 1 2): ").strip()
        if entrada.lower() == "fim":
            break
        try:
            gm, gs = map(int, entrada.split())
            team_b_games.append((gm, gs))
        except Exception as e:
            print("Entrada inválida. Tente novamente.", e)
    if not team_a_games or not team_b_games:
        print("Erro: Adicione pelo menos um jogo para cada time.")
        return None, None
    team_a = TeamStats(team_a_name, [gm for gm, gs in team_a_games], [gs for gm, gs in team_a_games])
    team_b = TeamStats(team_b_name, [gm for gm, gs in team_b_games], [gs for gm, gs in team_b_games])
    return team_a, team_b

def get_number_input(prompt):
    while True:
        entrada = input(prompt)
        try:
            number = float(entrada)
            return number
        except:
            print("Por favor, insira um número válido.")

# =============================================================================
# NOVAS FUNCIONALIDADES
# =============================================================================

def simulate_weather_impact(team, weather_factor=0.9):
    new_avg = team.average_goals_scored() * weather_factor
    print(f"\n🌦️ Impacto do Tempo para {team.name}:")
    print(f"Média original de gols: {team.average_goals_scored():.2f}")
    print(f"Média ajustada (com impacto do tempo): {new_avg:.2f}")
    return new_avg

def simulate_fatigue_impact(team, fatigue_factor=0.85):
    new_avg = team.average_goals_scored() * fatigue_factor
    print(f"\n😓 Impacto da Fadiga para {team.name}:")
    print(f"Média original de gols: {team.average_goals_scored():.2f}")
    print(f"Média ajustada (com impacto da fadiga): {new_avg:.2f}")
    return new_avg

def simulate_tournament(teams, n_matches=38):
    standings = {team.name: {"Pontos": 0, "Saldo": 0} for team in teams}
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            team_a = teams[i]
            team_b = teams[j]
            gols_a, gols_b = simulate_match(team_a, team_b)
            if gols_a > gols_b:
                standings[team_a.name]["Pontos"] += 3
            elif gols_a < gols_b:
                standings[team_b.name]["Pontos"] += 3
            else:
                standings[team_a.name]["Pontos"] += 1
                standings[team_b.name]["Pontos"] += 1
            standings[team_a.name]["Saldo"] += (gols_a - gols_b)
            standings[team_b.name]["Saldo"] += (gols_b - gols_a)
    ranking = sorted(standings.items(), key=lambda x: (x[1]["Pontos"], x[1]["Saldo"]), reverse=True)
    ranking_str = "\n".join([f"{pos}. {team}: {stats['Pontos']} pontos, Saldo de gols: {stats['Saldo']}" 
                             for pos, (team, stats) in enumerate(ranking, 1)])
    print("Classificação Final do Torneio:")
    print(ranking_str)
    return ranking

def generate_detailed_match_report(team_a, team_b, match_result, additional_stats, filename="detailed_match_report.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Relatório Detalhado da Partida ===\n")
        f.write(f"Time Mandante: {team_a.name}\n")
        f.write(f"Time Visitante: {team_b.name}\n")
        f.write(f"Placar Simulado: {team_a.name} {match_result[0]} x {match_result[1]} {team_b.name}\n\n")
        f.write("=== Estatísticas Adicionais ===\n")
        for key, value in additional_stats.items():
            f.write(f"{key}: {value}\n")
    print("Relatório detalhado salvo em", filename)

def export_simulation_to_pdf(data, filename="simulation_report.pdf"):
    try:
        from fpdf import FPDF
    except ImportError:
        print("⚠️ fpdf não está instalado. Instale com 'pip install fpdf'")
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Simulação de Futebol", ln=True, align="C")
    pdf.ln(10)
    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.output(filename)
    print("Relatório exportado para PDF:", filename)

def interactive_data_analysis(df):
    print("Iniciando análise interativa dos dados históricos...")
    teams = df["HomeTeam"].unique()
    for team in teams:
        team_data = df[df["HomeTeam"] == team]
        goals = team_data["FTHG"].tolist()
        plt.hist(goals, bins=range(0, max(goals)+2), alpha=0.7, label=team)
    plt.title("Distribuição de Gols Marcados em Casa")
    plt.xlabel("Gols")
    plt.ylabel("Frequência")
    plt.legend()
    plt.show()

# =============================================================================
# FUNÇÕES AUXILIARES PARA SIMULAÇÃO AVANÇADA
# =============================================================================

def adjust_offensive_defensive_factors(team):
    # Retorna os fatores básicos de ataque e defesa usando a função importar do módulo calcular_fatores
    return ajustar_fatores(team, "ataque", True), ajustar_fatores(team, "defesa", True)

def simulate_match_with_variation(team_a, team_b, base_confidence=1.0, seed=None):
    """
    Simula uma partida entre dois times com variação nos fatores ofensivos e defensivos.
    Esta função adiciona ruído aos fatores para simulação avançada.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    sigma_attack = 0.3
    sigma_defense = 0.3
    base_attack_a, base_defense_a = adjust_offensive_defensive_factors(team_a)
    base_attack_b, base_defense_b = adjust_offensive_defensive_factors(team_b)
    attack_a = base_attack_a + np.random.normal(0, sigma_attack)
    defense_a = base_defense_a + np.random.normal(0, sigma_defense)
    attack_b = base_attack_b + np.random.normal(0, sigma_attack)
    defense_b = base_defense_b + np.random.normal(0, sigma_defense)
    avg_goals_a = team_a.average_goals_scored() * attack_a
    avg_goals_b = team_b.average_goals_scored() * attack_b
    avg_goals_a *= max(0.5, 1 - team_b.average_goals_conceded() * defense_a)
    avg_goals_b *= max(0.5, 1 - team_a.average_goals_conceded() * defense_b)
    avg_goals_a = max(0.5, min(avg_goals_a, 5))
    avg_goals_b = max(0.5, min(avg_goals_b, 5))
    confidence = max(0.8, min(base_confidence + np.random.normal(0, 0.05), 1.2))
    gols_a = np.random.poisson(avg_goals_a * confidence)
    gols_b = np.random.poisson(avg_goals_b * confidence)
    return gols_a, gols_b

def validar_simulacoes(team, num_simulacoes=1000):
    simulated_gols = [np.random.poisson(team.average_goals_scored()) for _ in range(num_simulacoes)]
    media_simulada = np.mean(simulated_gols)
    desvio_simulado = np.std(simulated_gols)
    media_historica = team.average_goals_scored()
    desvio_historico = np.std(team.golsMarcados) if team.golsMarcados else 0
    print(f"\nEquipe: {team.name}\n"
          f"Média Histórica: {media_historica:.2f}, Média Simulada: {media_simulada:.2f}\n"
          f"Desvio Padrão Histórico: {desvio_historico:.2f}, Desvio Simulado: {desvio_simulado:.2f}")

# =============================================================================
# MENU PRINCIPAL – INTERAÇÃO VIA CONSOLE
# =============================================================================

def main_console():
    data = None
    team_a = None
    team_b = None
    while True:
        print("\n=== Sistema de Simulação de Futebol ===")
        print("1. Carregar Dados")
        print("2. Escolher Times")
        print("3. Simular Partida")
        print("4. Comparar Times")
        print("5. Visualizar Tendências")
        print("6. Gerar Relatório")
        print("7. Registrar Resultado de Simulação")
        print("8. Exibir Registros de Simulações")
        print("9. Ranking de Força (Gols)")
        print("10. Probabilidades de Mercado (Over/Under)")
        print("11. Previsão com Distribuição de Poisson")
        print("12. Previsão com Distribuição Skellam")
        print("13. Atualizar Rating Elo")
        print("14. Previsão via Regressão (Placeholder)")
        print("15. Simular Impacto do Tempo")
        print("16. Simular Impacto da Fadiga")
        print("17. Simular Torneio")
        print("18. Gerar Relatório Detalhado")
        print("19. Exportar Relatório para PDF")
        print("20. Análise Interativa dos Dados")
        print("21. Validar Simulações (Históricos)")
        print("0. Sair")
        opcao = input("Escolha uma opção: ").strip()

        if opcao == "0":
            print("Programa encerrado.")
            break
        elif opcao == "1":
            arquivos = input("Digite os caminhos dos arquivos Excel, separados por ponto e vírgula: ")
            if arquivos:
                file_list = [s.strip() for s in arquivos.split(";")]
                try:
                    data = carregar_dados_excel(file_list)
                    print("Dados carregados com sucesso!")
                except Exception as e:
                    print("Erro ao carregar dados:", e)
        elif opcao == "2":
            res = get_teams_console()
            if res is not None:
                team_a, team_b = res
                print(f"Times Selecionados: Time A: {team_a.name} | Time B: {team_b.name}")
        elif opcao == "3":
            if team_a is None or team_b is None:
                print("Erro: Escolha os times primeiro.")
            else:
                gols_a, gols_b = simulate_match(team_a, team_b)
                exp_a, exp_b = compute_basic_expected_values(team_a, team_b)
                prob_score = compute_score_probability(exp_a, exp_b, gols_a, gols_b)
                print(f"Placar Simulado: {team_a.name} {gols_a} x {gols_b} {team_b.name}")
                print(f"Probabilidade: {prob_score*100:.2f}%")
        elif opcao == "4":
            if team_a is None or team_b is None:
                print("Erro: Escolha os times primeiro.")
            else:
                backup = sys.stdout
                sys.stdout = mystdout = StringIO()
                comparar_times(team_a, team_b)
                sys.stdout = backup
                result = mystdout.getvalue()
                print("Comparação de Times:")
                print(result)
        elif opcao == "5":
            if team_a is None or team_b is None:
                print("Erro: Escolha os times primeiro.")
            else:
                plotar_tendencias(team_a, team_b)
        elif opcao == "6":
            if team_a is None or team_b is None or data is None:
                print("Erro: Certifique-se de ter carregado os dados e escolhido os times.")
            else:
                gols_a, gols_b = simulate_match(team_a, team_b)
                exp_a, exp_b = compute_basic_expected_values(team_a, team_b)
                prob_score = compute_score_probability(exp_a, exp_b, gols_a, gols_b)
                prob_market = calculate_goal_market_probabilities(team_a, team_b)
                h2h = calculate_h2h_statistics(team_a, team_b, data['dados_partidas'])
                gerar_relatorio(team_a, team_b, h2h, prob_market,
                                {'gols_a': gols_a, 'gols_b': gols_b},
                                team_a.golsMarcados[-5:], team_b.golsMarcados[-5:],
                                {"home": team_a.average_goals_scored(), "away": team_a.average_goals_conceded()},
                                {"home": team_b.average_goals_scored(), "away": team_b.average_goals_conceded()})
                print("Relatório gerado!")
        elif opcao == "7":
            registrar_resultado_simulacao()
        elif opcao == "8":
            exibir_registros_simulacoes()
        elif opcao == "9":
            if data is None:
                print("Erro: Carregue os dados primeiro.")
            else:
                gerar_ranking(data['dados_partidas'])
        elif opcao == "10":
            if team_a is None or team_b is None:
                print("Erro: Escolha os times primeiro.")
            else:
                prob_market = calculate_goal_market_probabilities(team_a, team_b)
                plotar_probabilidades_mercado(prob_market)
        elif opcao == "11":
            media_total = input("Informe a média de gols esperada (λ): ")
            max_goals = input("Informe o número máximo de gols: ")
            try:
                media_total = float(media_total)
                max_goals = int(max_goals)
            except:
                print("Valores inválidos.")
                continue
            probs = poisson_distribution_probabilities(media_total, max_goals)
            result = "\n".join([f"Golos = {k}: {prob*100:.2f}%" for k, prob in probs.items()])
            print("Distribuição de Poisson:")
            print(result)
        elif opcao == "12":
            lambda_a = input("Informe o λ do time A: ")
            lambda_b = input("Informe o λ do time B: ")
            diff = input("Informe a diferença de gols desejada: ")
            try:
                lambda_a = float(lambda_a)
                lambda_b = float(lambda_b)
                diff = int(diff)
            except:
                print("Valores inválidos.")
                continue
            prob = skellam_distribution_probability(lambda_a, lambda_b, diff)
            print(f"Probabilidade de uma diferença de {diff} gols: {prob*100:.2f}%")
        elif opcao == "13":
            current_rating = input("Informe o rating atual do time: ")
            opponent_rating = input("Informe o rating do adversário: ")
            result_val = input("Resultado (1 para vitória, 0.5 para empate, 0 para derrota): ")
            try:
                current_rating = float(current_rating)
                opponent_rating = float(opponent_rating)
                result_val = float(result_val)
            except:
                print("Valores inválidos.")
                continue
            new_rating = update_elo_rating(current_rating, opponent_rating, result_val)
            print(f"Novo rating atualizado: {new_rating:.2f}")
        elif opcao == "14":
            features = input("Informe os features separados por vírgula (ex.: 1.2,0.8,1.0): ")
            try:
                features = list(map(float, features.split(",")))
            except:
                print("Valores inválidos.")
                continue
            outcome_probs = predict_outcome_regression(features)
            result = "\n".join([f"{k}: {v*100:.2f}%" for k, v in outcome_probs.items()])
            print("Previsão via Regressão:")
            print(result)
        elif opcao == "15":
            if team_a is None:
                print("Erro: Escolha um time primeiro (utilize 'Escolher Times').")
            else:
                new_avg = simulate_weather_impact(team_a, weather_factor=0.9)
                print(f"Nova média ofensiva de {team_a.name} com impacto do tempo: {new_avg:.2f}")
        elif opcao == "16":
            if team_a is None:
                print("Erro: Escolha um time primeiro (utilize 'Escolher Times').")
            else:
                new_avg = simulate_fatigue_impact(team_a, fatigue_factor=0.85)
                print(f"Nova média ofensiva de {team_a.name} com impacto da fadiga: {new_avg:.2f}")
        elif opcao == "17":
            team_names = input("Digite os nomes dos times separados por vírgula: ")
            if team_names:
                names = [name.strip() for name in team_names.split(",")]
                teams = []
                for name in names:
                    # Cria times com resultados aleatórios para simulação
                    teams.append(TeamStats(name, [random.randint(0,4) for _ in range(5)], [random.randint(0,4) for _ in range(5)]))
                ranking = simulate_tournament(teams)
                ranking_str = "\n".join([f"{pos}. {team} - Pontos: {stats['Pontos']}, Saldo: {stats['Saldo']}" for pos, (team, stats) in enumerate(ranking,1)])
                print("Classificação do Torneio:")
                print(ranking_str)
        elif opcao == "18":
            if team_a is None or team_b is None:
                print("Erro: Escolha os times primeiro.")
            else:
                match_result = simulate_match(team_a, team_b)
                additional_stats = {
                    "Média de Gols Time A": team_a.average_goals_scored(),
                    "Média de Gols Time B": team_b.average_goals_scored(),
                    "Fator de Defesa Time A": team_a.average_goals_conceded(),
                    "Fator de Defesa Time B": team_b.average_goals_conceded()
                }
                generate_detailed_match_report(team_a, team_b, match_result, additional_stats)
                print("Relatório detalhado gerado!")
        elif opcao == "19":
            report_data = {
                "Time A": team_a.name if team_a else "N/A",
                "Time B": team_b.name if team_b else "N/A",
                "Média Gols Time A": team_a.average_goals_scored() if team_a else 0,
                "Média Gols Time B": team_b.average_goals_scored() if team_b else 0
            }
            export_simulation_to_pdf(report_data)
        elif opcao == "20":
            if data is None:
                print("Erro: Carregue os dados primeiro.")
            else:
                interactive_data_analysis(data['dados_partidas'])
        elif opcao == "21":
            if team_a is None:
                print("Erro: Escolha o time primeiro.")
            else:
                validar_simulacoes(team_a, num_simulacoes=1000)
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main_console()
