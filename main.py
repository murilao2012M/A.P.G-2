#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Simulação de Futebol – Versão Refatorada com Interface Gráfica
Melhorias:
  • Utiliza PySimpleGUI para interações com menus gráficos (não mais via console)
  • Os menus (abas) são definidos de forma estática, sem a necessidade de digitar eventos
  • Inclusão de novas funcionalidades: impacto do tempo, impacto da fadiga, simulação de torneio, etc.
  • O código original foi mantido integralmente (nenhuma linha removida, apenas adaptada)
  
OBS: A experiência visual se dará por meio da interface gráfica do PySimpleGUI e dos gráficos gerados pelo matplotlib.
"""

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
import PySimpleGUI as sg
import sys
from io import StringIO

# Importa funções dos módulos (assegure-se de que esses módulos estejam na pasta "modules")
from modules import gerar_ranking, gerar_relatorio
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
# FUNÇÕES ORIGINAIS
# =============================================================================

def save_to_excel(data, filename="registros_jogos.xlsx"):
    try:
        existing_data = pd.read_excel(filename)
        new_data = pd.concat([existing_data, data], ignore_index=True).drop_duplicates()
    except FileNotFoundError:
        new_data = data
    new_data.to_excel(filename, index=False)
    sg.popup("✅ Dados atualizados em", filename)

def mostrar_escudo(time):
    caminho_escudo = f'escudos/{time}.png'
    if os.path.exists(caminho_escudo):
        img = Image.open(caminho_escudo)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Escudo do {time}", fontsize=14, weight='bold')
        plt.show()
    else:
        sg.popup_error(f"⚠️ Escudo do {time} não encontrado.")

class TeamStats:
    def __init__(self, name, golsMarcados, golsSofridos):
        if not isinstance(golsMarcados, list) or not isinstance(golsSofridos, list):
            raise ValueError("golsMarcados e golsSofridos devem ser listas.")
        self.name = name
        self.golsMarcados = golsMarcados
        self.golsSofridos = golsSofridos

    def average_goals_scored(self):
        if not self.golsMarcados:
            sg.popup_error(f"⚠️ Não há dados de gols marcados para {self.name}.")
            return 0
        return sum(self.golsMarcados) / len(self.golsMarcados)

    def average_goals_conceded(self):
        if not self.golsSofridos:
            sg.popup_error(f"⚠️ Não há dados de gols sofridos para {self.name}.")
            return 0
        return sum(self.golsSofridos) / len(self.golsSofridos)

    def last_n_game_performance(self, n=5):
        n = min(n, len(self.golsMarcados), len(self.golsSofridos))
        if n == 0:
            sg.popup_error(f"⚠️ Dados insuficientes para {self.name}.")
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
            sg.popup_error("⚠️ Dados insuficientes para previsão de gols.")
            return 0
        mean_gols = (self.average_goals_scored() + time_adversario.average_goals_conceded()) / 2
        return np.random.poisson(mean_gols)

    def simular_partida_monte_carlo(self, time_adversario, n_simulacoes=5000):
        try:
            resultados = [simulate_match(self, time_adversario) for _ in range(n_simulacoes)]
        except Exception as e:
            sg.popup_error(f"Erro na simulação de partida: {e}")
            return 0, 0, 0
        vitorias = sum(1 for g_a, g_b in resultados if g_a > g_b)
        empates = sum(1 for g_a, g_b in resultados if g_a == g_b)
        derrotas = sum(1 for g_a, g_b in resultados if g_a < g_b)
        prob_vitoria = (vitorias / n_simulacoes) * 100
        prob_empate = (empates / n_simulacoes) * 100
        prob_derrota = (derrotas / n_simulacoes) * 100
        sg.popup(f"Probabilidades ({n_simulacoes} simulações):\n"
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
        sg.popup(f"✅ Resultado adicionado para {self.name}: {gols_marcados} marcados, {gols_sofridos} sofridos.")

    def average_goals_scored_weighted(self, weight_factor=0.9):
        if not self.golsMarcados:
            sg.popup_error(f"⚠️ Não há dados para {self.name}.")
            return 0
        pesos = [weight_factor ** i for i in range(len(self.golsMarcados))]
        pesos.reverse()
        total_pesos = sum(pesos)
        return sum(g * p for g, p in zip(self.golsMarcados, pesos)) / total_pesos

    def average_goals_conceded_weighted(self, weight_factor=0.9):
        if not self.golsSofridos:
            sg.popup_error(f"⚠️ Não há dados para {self.name}.")
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
    sg.popup("=== RandomForest ===",
             f"Acurácia (hold-out): {accuracy_score(y_test, y_pred):.2f}",
             "Relatório de classificação:",
             classification_report(y_test, y_pred))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    sg.popup("Acurácia Cross-Val (5 folds):", f"{round(scores.mean(), 3)} +/- {round(scores.std(), 3)}")
    return model

def train_model_xgboost(df):
    X = df[['Gols Marcados', 'Gols Sofridos', 'Vitórias', 'Empates', 'Derrotas']]
    y = df['Resultado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    sg.popup("=== XGBoost ===",
             f"Acurácia (hold-out): {accuracy_score(y_test, y_pred):.2f}",
             "Relatório de classificação:",
             classification_report(y_test, y_pred))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')
    sg.popup("Acurácia Cross-Val (5 folds):", f"{round(scores.mean(), 3)} +/- {round(scores.std(), 3)}")
    return xgb_model

# =============================================================================
# REGISTRO E EXPORTAÇÃO DE SIMULAÇÕES
# =============================================================================

simulation_records = []

def registrar_resultado_simulacao():
    time_a = sg.popup_get_text("Informe o nome do Time A:")
    time_b = sg.popup_get_text("Informe o nome do Time B:")
    placar = sg.popup_get_text("Informe o placar (ex.: 2-1):")
    mercado = sg.popup_get_text("Informe o mercado escolhido:")
    ganhadora = sg.popup_get_text("Essa simulação foi ganhadora? (s/n):").lower()
    registro = {
        "Time A": time_a,
        "Time B": time_b,
        "Placar": placar,
        "Mercado": mercado,
        "Ganhadora": "Sim" if ganhadora in ["s", "sim"] else "Não"
    }
    simulation_records.append(registro)
    sg.popup("✅ Resultado registrado com sucesso!")

def exibir_registros_simulacoes():
    if not simulation_records:
        sg.popup("⚠️ Nenhum registro de simulação encontrado.")
        return
    registros = "\n".join([f"{i+1}. Time A: {reg['Time A']} | Time B: {reg['Time B']} | Placar: {reg['Placar']} | Mercado: {reg['Mercado']} | Ganhadora: {reg['Ganhadora']}"
                           for i, reg in enumerate(simulation_records)])
    sg.popup_scrolled(registros, title="Registros de Simulações")

def export_simulation_report(data, filename="relatorio_simulacoes.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    sg.popup("✅ Relatório exportado para", filename)

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
# FUNÇÃO DE ENTRADA DOS DADOS DOS TIMES (Jogo a Jogo ou Totais)
# =============================================================================

def get_teams_gui():
    tab_team_a = [
        [sg.Text("Nome do Time A:", font=("Helvetica", 14)),
         sg.Input(key="-TEAM_A_NAME-", font=("Helvetica", 14), size=(20,1))],
        [sg.Text("Adicionar Jogo (Gols Marcados / Gols Sofridos):", font=("Helvetica", 12))],
        [sg.Text("Marcados:"), sg.Spin(values=list(range(0,11)), initial_value=0, key="-TEAM_A_GM-", font=("Helvetica", 14), size=(5,1)),
         sg.Text("Sofridos:"), sg.Spin(values=list(range(0,11)), initial_value=0, key="-TEAM_A_GS-", font=("Helvetica", 14), size=(5,1)),
         sg.Button("Adicionar Jogo", key="-TEAM_A_ADD-", font=("Helvetica", 12))],
        [sg.Listbox(values=[], key="-TEAM_A_LIST-", size=(30,5), font=("Helvetica", 12))]
    ]
    tab_team_b = [
        [sg.Text("Nome do Time B:", font=("Helvetica", 14)),
         sg.Input(key="-TEAM_B_NAME-", font=("Helvetica", 14), size=(20,1))],
        [sg.Text("Adicionar Jogo (Gols Marcados / Gols Sofridos):", font=("Helvetica", 12))],
        [sg.Text("Marcados:"), sg.Spin(values=list(range(0,11)), initial_value=0, key="-TEAM_B_GM-", font=("Helvetica", 14), size=(5,1)),
         sg.Text("Sofridos:"), sg.Spin(values=list(range(0,11)), initial_value=0, key="-TEAM_B_GS-", font=("Helvetica", 14), size=(5,1)),
         sg.Button("Adicionar Jogo", key="-TEAM_B_ADD-", font=("Helvetica", 12))],
        [sg.Listbox(values=[], key="-TEAM_B_LIST-", size=(30,5), font=("Helvetica", 12))]
    ]
    layout = [
        [sg.TabGroup([[sg.Tab("Time A", tab_team_a), sg.Tab("Time B", tab_team_b)]], key="-TABGROUP-")],
        [sg.Button("Confirmar", key="Confirmar", font=("Helvetica", 14)),
         sg.Button("Cancelar", key="Cancelar", font=("Helvetica", 14))]
    ]
    window = sg.Window("Dados dos Times", layout, size=(800,480), finalize=True)
    team_a_games = []
    team_b_games = []
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Cancelar"):
            window.close()
            return None, None
        if event == "-TEAM_A_ADD-":
            gm = int(values["-TEAM_A_GM-"])
            gs = int(values["-TEAM_A_GS-"])
            team_a_games.append((gm, gs))
            window["-TEAM_A_LIST-"].update(team_a_games)
        if event == "-TEAM_B_ADD-":
            gm = int(values["-TEAM_B_GM-"])
            gs = int(values["-TEAM_B_GS-"])
            team_b_games.append((gm, gs))
            window["-TEAM_B_LIST-"].update(team_b_games)
        if event == "Confirmar":
            try:
                team_a_name = values["-TEAM_A_NAME-"]
                team_b_name = values["-TEAM_B_NAME-"]
                if not team_a_name or not team_b_name:
                    sg.popup_error("Preencha os nomes dos times.")
                    continue
                if not team_a_games or not team_b_games:
                    sg.popup_error("Adicione pelo menos um jogo para cada time.")
                    continue
            except Exception as e:
                sg.popup_error("Erro ao processar dados: " + str(e))
                continue
            team_a = TeamStats(team_a_name, [gm for gm, gs in team_a_games], [gs for gm, gs in team_a_games])
            team_b = TeamStats(team_b_name, [gm for gm, gs in team_b_games], [gs for gm, gs in team_b_games])
            window.close()
            return team_a, team_b

def get_number_input(prompt, key="NUMBER"):
    layout = [
        [sg.Text(prompt, font=("Helvetica", 14))],
        [sg.Input(key=key, font=("Helvetica", 14))],
        [sg.Button("OK", font=("Helvetica", 14)), sg.Button("Cancelar", font=("Helvetica", 14))]
    ]
    window = sg.Window("Entrada Numérica", layout, size=(400,200))
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Cancelar"):
            window.close()
            return None
        if event == "OK":
            try:
                number = float(values[key])
                window.close()
                return number
            except:
                sg.popup_error("Por favor, insira um número válido.")

# =============================================================================
# NOVAS FUNCIONALIDADES
# =============================================================================

def simulate_weather_impact(team, weather_factor=0.9):
    new_avg = team.average_goals_scored() * weather_factor
    sg.popup(f"\n🌦️ Impacto do Tempo para {team.name}:\n"
             f"Média original de gols: {team.average_goals_scored():.2f}\n"
             f"Média ajustada (com impacto do tempo): {new_avg:.2f}")
    return new_avg

def simulate_fatigue_impact(team, fatigue_factor=0.85):
    new_avg = team.average_goals_scored() * fatigue_factor
    sg.popup(f"\n😓 Impacto da Fadiga para {team.name}:\n"
             f"Média original de gols: {team.average_goals_scored():.2f}\n"
             f"Média ajustada (com impacto da fadiga): {new_avg:.2f}")
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
    sg.popup_scrolled(ranking_str, title="Classificação Final do Torneio")
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
    sg.popup("Relatório detalhado salvo em", filename)

def export_simulation_to_pdf(data, filename="simulation_report.pdf"):
    try:
        from fpdf import FPDF
    except ImportError:
        sg.popup_error("⚠️ fpdf não está instalado. Instale com 'pip install fpdf'")
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Simulação de Futebol", ln=True, align="C")
    pdf.ln(10)
    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.output(filename)
    sg.popup("Relatório exportado para PDF:", filename)

def interactive_data_analysis(df):
    sg.popup("Iniciando análise interativa dos dados históricos...")
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
    sg.popup(f"\nEquipe: {team.name}\n"
             f"Média Histórica: {media_historica:.2f}, Média Simulada: {media_simulada:.2f}\n"
             f"Desvio Padrão Histórico: {desvio_historico:.2f}, Desvio Simulado: {desvio_simulado:.2f}")

# =============================================================================
# MENU PRINCIPAL – LAYOUT E EVENT LOOP
# =============================================================================

def main_gui_updated():
    sg.theme("DarkBlue3")
    
    # Layout para cada aba
    tab_config_layout = [
        [sg.Text("CONFIGURAÇÃO", font=("Helvetica", 16, "bold"), justification="center", expand_x=True)],
        [sg.Button("Carregar Dados", key="Carregar_Dados", size=(20,1), font=("Helvetica", 14)),
         sg.Button("Escolher Times", key="Escolher_Times", size=(20,1), font=("Helvetica", 14))],
        [sg.Text("Status dos Dados:"), sg.Text("", size=(40,1), key="Data_Status", font=("Helvetica", 14))],
        [sg.Text("Times Selecionados:"), sg.Text("", size=(40,1), key="Team_Status", font=("Helvetica", 14))]
    ]
    
    tab_simulacao_layout = [
        [sg.Text("SIMULAÇÃO", font=("Helvetica", 16, "bold"), justification="center", expand_x=True)],
        [sg.Button("Simular Partida", key="Simular_Partida", size=(25,1), font=("Helvetica", 14)),
         sg.Button("Simulação Avançada (Monte Carlo)", key="Simulacao_MC", size=(25,1), font=("Helvetica", 14))],
        [sg.Button("Simulação Avançada com Variação", key="Simulacao_Variacao", size=(25,1), font=("Helvetica", 14)),
         sg.Button("Simular Temporada", key="Simular_Temporada", size=(25,1), font=("Helvetica", 14))],
        [sg.Button("Simular Impacto de Lesões", key="Simular_Lesao", size=(25,1), font=("Helvetica", 14))]
    ]
    
    tab_analise_layout = [
        [sg.Text("ANÁLISES", font=("Helvetica", 16, "bold"), justification="center", expand_x=True)],
        [sg.Button("Comparar Times", key="Comparar_Times", size=(25,1), font=("Helvetica", 14)),
         sg.Button("Visualizar Tendências", key="Visualizar_Tendencias", size=(25,1), font=("Helvetica", 14))],
        [sg.Button("Validar Simulações (Históricos)", key="Validar_Simulacoes", size=(25,1), font=("Helvetica", 14)),
         sg.Button("Exibir XG (Expected Goals)", key="Exibir_XG", size=(25,1), font=("Helvetica", 14))],
        [sg.Button("Análise de Sensibilidade", key="Analise_Sensibilidade", size=(25,1), font=("Helvetica", 14))]
    ]
    
    tab_relatorios_layout = [
        [sg.Text("RELATÓRIOS & REGISTROS", font=("Helvetica", 16, "bold"), justification="center", expand_x=True)],
        [sg.Button("Gerar Relatório", key="Gerar_Relatorio", size=(25,1), font=("Helvetica", 14)),
         sg.Button("Exportar Relatório", key="Exportar_Relatorio", size=(25,1), font=("Helvetica", 14))],
        [sg.Button("Registrar Resultado de Simulação", key="Registrar_Resultado", size=(25,1), font=("Helvetica", 14)),
         sg.Button("Exibir Registros de Simulações", key="Exibir_Registros", size=(25,1), font=("Helvetica", 14))],
        [sg.Button("Ranking de Força (Gols)", key="Ranking_Forca", size=(25,1), font=("Helvetica", 14))]
    ]
    
    tab_previsoes_layout = [
        [sg.Text("PREVISÕES", font=("Helvetica", 16, "bold"), justification="center", expand_x=True)],
        [sg.Button("Probabilidades de Mercado (Over/Under)", key="Prob_Mercado", size=(30,1), font=("Helvetica", 14))],
        [sg.Button("Previsão com Distribuição de Poisson", key="Prev_Poisson", size=(30,1), font=("Helvetica", 14)),
         sg.Button("Previsão com Distribuição Skellam", key="Prev_Skellam", size=(30,1), font=("Helvetica", 14))],
        [sg.Button("Atualizar Rating Elo", key="Atualizar_Elo", size=(30,1), font=("Helvetica", 14)),
         sg.Button("Previsão via Regressão (Placeholder)", key="Prev_Regressao", size=(30,1), font=("Helvetica", 14))]
    ]
    
    tab_novas_layout = [
        [sg.Text("NOVAS FUNCIONALIDADES", font=("Helvetica", 16, "bold"), justification="center", expand_x=True)],
        [sg.Button("Simular Impacto do Tempo", key="Simular_Tempo", size=(25,1), font=("Helvetica", 14),
                   tooltip="Simula o impacto das condições climáticas no desempenho do time")],
        [sg.Button("Simular Impacto da Fadiga", key="Simular_Fadiga", size=(25,1), font=("Helvetica", 14),
                   tooltip="Simula o impacto da fadiga no desempenho do time")],
        [sg.Button("Simular Torneio", key="Simular_Torneio", size=(25,1), font=("Helvetica", 14),
                   tooltip="Simula um torneio round-robin entre vários times")],
        [sg.Button("Gerar Relatório Detalhado", key="Gerar_Relatorio_Detalhado", size=(25,1), font=("Helvetica", 14),
                   tooltip="Gera um relatório detalhado da última partida simulada")],
        [sg.Button("Exportar Relatório para PDF", key="Exportar_PDF", size=(25,1), font=("Helvetica", 14),
                   tooltip="Exporta um relatório de simulação para PDF")],
        [sg.Button("Análise Interativa dos Dados", key="Analise_Interativa", size=(25,1), font=("Helvetica", 14),
                   tooltip="Visualiza gráficos interativos dos dados históricos")]
    ]
    
    layout = [
        [sg.TabGroup([[sg.Tab("Configuração", tab_config_layout),
                       sg.Tab("Simulação", tab_simulacao_layout),
                       sg.Tab("Análises", tab_analise_layout),
                       sg.Tab("Relatórios & Registros", tab_relatorios_layout),
                       sg.Tab("Previsões", tab_previsoes_layout),
                       sg.Tab("Novas Funcionalidades", tab_novas_layout)]], key="-TABGROUP-", expand_x=True, expand_y=True)],
        [sg.Button("Sair", key="Sair", size=(10,1), font=("Helvetica", 14))]
    ]
    
    window = sg.Window("Sistema de Simulação de Futebol", layout, size=(900,600), finalize=True, resizable=True)
        
    # Variáveis de controle
    data = None
    team_a = None
    team_b = None
    simulation_records = []
    
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Sair"):
            break
        
        # Aba Configuração
        if event == "Carregar_Dados":
            files = sg.popup_get_text("Digite os caminhos dos arquivos Excel, separados por ponto e vírgula:")
            if files:
                file_list = files.split(";")
                try:
                    data = carregar_dados_excel(file_list)
                    window["Data_Status"].update("Dados carregados com sucesso!")
                except Exception as e:
                    sg.popup_error("Erro ao carregar dados: " + str(e))
        elif event == "Escolher_Times":
            res = get_teams_gui()
            if res is not None:
                team_a, team_b = res
                window["Team_Status"].update(f"Time A: {team_a.name} | Time B: {team_b.name}")
        
        # Aba Simulação
        elif event == "Simular_Partida":
            if team_a is None or team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                gols_a, gols_b = simulate_match(team_a, team_b)
                exp_a, exp_b = compute_basic_expected_values(team_a, team_b)
                prob_score = compute_score_probability(exp_a, exp_b, gols_a, gols_b)
                sg.popup(f"Placar Simulado:\n{team_a.name} {gols_a} x {gols_b} {team_b.name}\nProbabilidade: {prob_score*100:.2f}%")
                
        elif event == "Comparar_Times":
            if team_a is None or team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                backup = sys.stdout
                sys.stdout = mystdout = StringIO()
                comparar_times(team_a, team_b)
                sys.stdout = backup
                result = mystdout.getvalue()
                sg.popup_scrolled(result, title="Comparação de Times")
                
        elif event in ("Visualizar_Tendencias", "Visualizar_Tendências"):
            if team_a is None or team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                plotar_tendencias(team_a, team_b)
                
        elif event in ("Configurar_Cenario_Personalizado", "Configurar Cenário Personalizado"):
            if team_a is None or team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                simular_cenario(team_a, team_b)
                
        elif event == "Gerar_Relatorio":
            if team_a is None or team_b is None or data is None:
                sg.popup_error("Certifique-se de ter carregado os dados e escolhido os times.")
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
                sg.popup("Relatório gerado!")
                
        elif event in ("Ver_Probabilidades_de_Mercado_(Over/Under)", "Prob_Mercado"):
            if team_a is None or team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                prob_market = calculate_goal_market_probabilities(team_a, team_b)
                plotar_probabilidades_mercado(prob_market)
                
        elif event == "Ranking_Forca":
            if data is None:
                sg.popup_error("Carregue os dados primeiro.")
            else:
                gerar_ranking(data['dados_partidas'])
                
        elif event == "Exibir_XG":
            if team_a is None or team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                exibir_xg(team_a, team_b)
                
        elif event == "Calcular_Recorrencia_de_Gols":
            if team_a is None and team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                gols_input = sg.popup_get_text("Informe o número de gols para calcular a recorrência:")
                if gols_input:
                    try:
                        gols = int(gols_input)
                    except:
                        sg.popup_error("Valor inválido.")
                        continue
                    qual_time = sg.popup_get_text("Escolha o time para cálculo (A/B):").upper()
                    if qual_time == "A":
                        recorrencia = calcular_recorrencia_gols(gols, team_a.golsMarcados)
                    else:
                        recorrencia = calcular_recorrencia_gols(gols, team_b.golsMarcados)
                    sg.popup(f"Probabilidade de {gols} gols: {recorrencia*100:.2f}%")
                    
        elif event == "Simulacao_MC":
            if team_a is None or team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                team_a.simular_partida_monte_carlo(team_b, n_simulacoes=50)
                
        elif event == "Simulacao_Variacao":
            if team_a is None or team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                gols_a, gols_b = simulate_match_with_variation(team_a, team_b, base_confidence=1.0)
                exp_a, exp_b = compute_basic_expected_values(team_a, team_b)
                prob_score = compute_score_probability(exp_a, exp_b, gols_a, gols_b)
                sg.popup(f"Simulação Avançada com Variação:\n{team_a.name} {gols_a} x {team_b.name} {gols_b}\nProbabilidade: {prob_score*100:.2f}%")
                
        elif event == "Validar_Simulacoes":
            if team_a is None:
                sg.popup_error("Escolha o time primeiro.")
            else:
                validar_simulacoes(team_a, num_simulacoes=1000)
                
        elif event == "Registrar_Resultado":
            layout_reg = [
                [sg.Text("Time A:", font=("Helvetica", 14)), sg.Input(key="-REG_TEAM_A-", font=("Helvetica", 14))],
                [sg.Text("Time B:", font=("Helvetica", 14)), sg.Input(key="-REG_TEAM_B-", font=("Helvetica", 14))],
                [sg.Text("Placar (ex.: 2-1):", font=("Helvetica", 14)), sg.Input(key="-REG_SCORE-", font=("Helvetica", 14))],
                [sg.Text("Mercado escolhido:", font=("Helvetica", 14)), sg.Input(key="-REG_MARKET-", font=("Helvetica", 14))],
                [sg.Text("Simulação ganhadora? (s/n):", font=("Helvetica", 14)), sg.Input(key="-REG_WIN-", font=("Helvetica", 14))],
                [sg.Button("Registrar", font=("Helvetica", 14)), sg.Button("Cancelar", font=("Helvetica", 14))]
            ]
            window_reg = sg.Window("Registrar Resultado de Simulação", layout_reg, size=(600,300))
            event_reg, values_reg = window_reg.read()
            if event_reg == "Registrar":
                registro = {
                    "Time A": values_reg["-REG_TEAM_A-"],
                    "Time B": values_reg["-REG_TEAM_B-"],
                    "Placar": values_reg["-REG_SCORE-"],
                    "Mercado": values_reg["-REG_MARKET-"],
                    "Ganhadora": "Sim" if values_reg["-REG_WIN-"].strip().lower() in ["s", "sim"] else "Não"
                }
                simulation_records.append(registro)
                sg.popup("Resultado registrado com sucesso!")
            window_reg.close()
                
        elif event == "Exibir_Registros":
            if not simulation_records:
                sg.popup("Nenhum registro de simulação encontrado.")
            else:
                registros = "\n".join([f"{i+1}. Time A: {reg['Time A']} | Time B: {reg['Time B']} | Placar: {reg['Placar']} | Mercado: {reg['Mercado']} | Ganhadora: {reg['Ganhadora']}" 
                                       for i, reg in enumerate(simulation_records)])
                sg.popup_scrolled(registros, title="Registros de Simulações")
                
        elif event == "Prev_Poisson":
            media_total = sg.popup_get_text("Informe a média de gols esperada (λ):", font=("Helvetica", 14))
            max_goals = sg.popup_get_text("Informe o número máximo de gols:", font=("Helvetica", 14))
            try:
                media_total = float(media_total)
                max_goals = int(max_goals)
            except:
                sg.popup_error("Valores inválidos.")
                continue
            probs = poisson_distribution_probabilities(media_total, max_goals)
            result = "\n".join([f"Golos = {k}: {prob*100:.2f}%" for k, prob in probs.items()])
            sg.popup_scrolled(result, title="Distribuição de Poisson")
                
        elif event == "Prev_Skellam":
            lambda_a = sg.popup_get_text("Informe o λ do time A:", font=("Helvetica", 14))
            lambda_b = sg.popup_get_text("Informe o λ do time B:", font=("Helvetica", 14))
            diff = sg.popup_get_text("Informe a diferença de gols desejada:", font=("Helvetica", 14))
            try:
                lambda_a = float(lambda_a)
                lambda_b = float(lambda_b)
                diff = int(diff)
            except:
                sg.popup_error("Valores inválidos.")
                continue
            prob = skellam_distribution_probability(lambda_a, lambda_b, diff)
            sg.popup(f"Probabilidade de uma diferença de {diff} gols: {prob*100:.2f}%")
                
        elif event == "Atualizar_Elo":
            current_rating = sg.popup_get_text("Informe o rating atual do time:", font=("Helvetica", 14))
            opponent_rating = sg.popup_get_text("Informe o rating do adversário:", font=("Helvetica", 14))
            result_val = sg.popup_get_text("Resultado (1 para vitória, 0.5 para empate, 0 para derrota):", font=("Helvetica", 14))
            try:
                current_rating = float(current_rating)
                opponent_rating = float(opponent_rating)
                result_val = float(result_val)
            except:
                sg.popup_error("Valores inválidos.")
                continue
            new_rating = update_elo_rating(current_rating, opponent_rating, result_val)
            sg.popup(f"Novo rating atualizado: {new_rating:.2f}")
                
        elif event == "Prev_Regressao":
            features = sg.popup_get_text("Informe os features separados por vírgula (ex.: 1.2,0.8,1.0):", font=("Helvetica", 14))
            try:
                features = list(map(float, features.split(",")))
            except:
                sg.popup_error("Valores inválidos.")
                continue
            outcome_probs = predict_outcome_regression(features)
            result = "\n".join([f"{k}: {v*100:.2f}%" for k, v in outcome_probs.items()])
            sg.popup_scrolled(result, title="Previsão via Regressão")
                
        # NOVAS FUNCIONALIDADES
        elif event == "Simular_Tempo":
            if team_a is None:
                sg.popup_error("Escolha um time primeiro (utilize 'Escolher Times').")
            else:
                new_avg = simulate_weather_impact(team_a, weather_factor=0.9)
                sg.popup(f"Nova média ofensiva de {team_a.name} com impacto do tempo: {new_avg:.2f}")
                
        elif event == "Simular_Fadiga":
            if team_a is None:
                sg.popup_error("Escolha um time primeiro (utilize 'Escolher Times').")
            else:
                new_avg = simulate_fatigue_impact(team_a, fatigue_factor=0.85)
                sg.popup(f"Nova média ofensiva de {team_a.name} com impacto da fadiga: {new_avg:.2f}")
                
        elif event == "Simular_Torneio":
            team_names = sg.popup_get_text("Digite os nomes dos times separados por vírgula:")
            if team_names:
                names = [name.strip() for name in team_names.split(",")]
                teams = []
                for name in names:
                    # Cria times com resultados aleatórios para simulação
                    teams.append(TeamStats(name, [random.randint(0,4) for _ in range(5)], [random.randint(0,4) for _ in range(5)]))
                ranking = simulate_tournament(teams)
                ranking_str = "\n".join([f"{pos}. {team} - Pontos: {stats['Pontos']}, Saldo: {stats['Saldo']}" for pos, (team, stats) in enumerate(ranking,1)])
                sg.popup_scrolled(ranking_str, title="Classificação do Torneio")
                
        elif event == "Gerar_Relatorio_Detalhado":
            if team_a is None or team_b is None:
                sg.popup_error("Escolha os times primeiro.")
            else:
                match_result = simulate_match(team_a, team_b)
                additional_stats = {
                    "Média de Gols Time A": team_a.average_goals_scored(),
                    "Média de Gols Time B": team_b.average_goals_scored(),
                    "Fator de Defesa Time A": team_a.average_goals_conceded(),
                    "Fator de Defesa Time B": team_b.average_goals_conceded()
                }
                generate_detailed_match_report(team_a, team_b, match_result, additional_stats)
                sg.popup("Relatório detalhado gerado!")
                
        elif event == "Exportar_PDF":
            report_data = {
                "Time A": team_a.name if team_a else "N/A",
                "Time B": team_b.name if team_b else "N/A",
                "Média Gols Time A": team_a.average_goals_scored() if team_a else 0,
                "Média Gols Time B": team_b.average_goals_scored() if team_b else 0
            }
            export_simulation_to_pdf(report_data)
                
        elif event == "Analise_Interativa":
            if data is None:
                sg.popup_error("Carregue os dados primeiro.")
            else:
                interactive_data_analysis(data['dados_partidas'])
                
    window.close()

if __name__ == "__main__":
    main_gui_updated()
    sg.popup("Programa encerrado.")
    exit()
