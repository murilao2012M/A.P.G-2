import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calcular_media_xg(team, n, ponderar_recente=False, prever_futuro=False, mostrar_grafico=False):
    """
    Calculates the average xG (expected goals) of the last 'n' games, with options for:
      - Weighted recent games
      - AI-based trend prediction
      - Anomaly detection (unexpected xG spikes)
      - Visualization of xG evolution

    Parameters:
        team (TeamStats): Object representing the team.
        n (int): Number of games to consider.
        ponderar_recente (bool): If True, applies **exponential weighting** to recent matches.
        prever_futuro (bool): If True, uses **Linear Regression AI** to predict future xG trends.
        mostrar_grafico (bool): If True, **plots xG evolution** with trend analysis.

    Returns:
        dict: Dictionary containing xG statistics, variation, trend, and anomalies.
    """
    try:
        # Check if xG data is available and sufficient
        if not hasattr(team, 'xg_data') or len(team.xg_data) < n:
            raise ValueError(f"Insufficient xG data for {team.name} in the last {n} games.")

        # Get the last 'n' xG values
        xg_list = team.xg_data[-n:]

        # Apply weighting to recent matches
        if ponderar_recente:
            weights = np.exp(np.linspace(0, 1, n))  # Exponential decay for recent games
            media_xg_ponderada = np.average(xg_list, weights=weights)
        else:
            media_xg_ponderada = np.mean(xg_list)  # Simple mean

        # Calculate xG variation trend
        xg_variation = (xg_list[-1] - xg_list[0]) / (xg_list[0] + 1e-6)  # Avoid division by zero

        # Identify anomalies (high and low xG performances)
        xg_std = np.std(xg_list)
        maior_xg = max(xg_list)
        menor_xg = min(xg_list)
        jogo_maior_xg = xg_list.index(maior_xg) + 1
        jogo_menor_xg = xg_list.index(menor_xg) + 1

        # Detect abnormal xG deviations
        anomalies = [(i+1, xg) for i, xg in enumerate(xg_list) if abs(xg - media_xg_ponderada) > 1.5 * xg_std]

        # AI-based Future xG Prediction (Linear Regression)
        future_xg_prediction = None
        if prever_futuro:
            x = np.arange(n).reshape(-1, 1)
            y = np.array(xg_list).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y)
            future_xg_prediction = float(model.predict([[n]]))  # Predict xG for next match

        # Generate xG Evolution Graph
        if mostrar_grafico:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, n+1), xg_list, marker='o', label='xG per Game', linestyle='-', color='blue')
            plt.axhline(y=media_xg_ponderada, color='green', linestyle='--', label=f"Average xG ({media_xg_ponderada:.2f})")
            if prever_futuro:
                plt.scatter(n+1, future_xg_prediction, color='red', label=f"Predicted xG ({future_xg_prediction:.2f})", marker='x')
            plt.xlabel("Last Games")
            plt.ylabel("Expected Goals (xG)")
            plt.title(f"xG Evolution for {team.name}")
            plt.legend()
            plt.grid()
            plt.show()

        # Return computed metrics
        return {
            "media_xg": media_xg_ponderada,
            "variacao_xg": xg_variation,
            "maior_xg": maior_xg,
            "menor_xg": menor_xg,
            "jogo_maior_xg": jogo_maior_xg,
            "jogo_menor_xg": jogo_menor_xg,
            "anomalies": anomalies,
            "future_xg": future_xg_prediction if prever_futuro else None
        }

    except ValueError as e:
        print(f"⚠️ {e}")
        return {}
    except AttributeError:
        print(f"⚠️ The team {team.name} has no xG data.")
        return {}