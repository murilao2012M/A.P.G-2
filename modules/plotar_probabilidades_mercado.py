from matplotlib import pyplot as plt

def plotar_probabilidades_mercado(probabilities, title="Probabilidades de Mercado", bar_width=0.4, custom_colors=None, display_values=True):
    """
    Plota um gráfico de barras para as probabilidades Over/Under com a possibilidade de personalizar cores, largura das barras e exibir os valores.
    
    :param probabilities: Dicionário com as probabilidades para cada mercado Over/Under.
    :param title: Título do gráfico.
    :param bar_width: Largura das barras do gráfico.
    :param custom_colors: Dicionário com as cores personalizadas para 'over' e 'under' (exemplo: {'over': 'green', 'under': 'red'}).
    :param display_values: Se True, exibe os valores percentuais nas barras.
    """
    
    # Verifica se o input probabilities tem o formato correto
    if not isinstance(probabilities, dict):
        raise ValueError("O parâmetro 'probabilities' deve ser um dicionário.")
    
    if not all(isinstance(v, dict) and 'over' in v and 'under' in v for v in probabilities.values()):
        raise ValueError("O dicionário 'probabilities' deve conter dicionários com as chaves 'over' e 'under'.")
    
    mercados = list(probabilities.keys())
    overs = [probabilities[mercado]['over'] * 100 for mercado in mercados]
    unders = [probabilities[mercado]['under'] * 100 for mercado in mercados]

    # Definindo cores customizadas, se fornecido
    color_over = custom_colors.get('over', 'blue') if custom_colors else 'blue'
    color_under = custom_colors.get('under', 'red') if custom_colors else 'red'

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(mercados))

    ax.bar(x, overs, width=bar_width, label='Over (%)', color=color_over, alpha=0.7)
    ax.bar([p + bar_width for p in x], unders, width=bar_width, label='Under (%)', color=color_under, alpha=0.7)

    # Adicionando valores nas barras, se display_values for True
    if display_values:
        for i in range(len(mercados)):
            ax.text(x[i], overs[i] + 2, f"{overs[i]:.1f}%", ha='center', va='bottom', fontsize=10, color='black')
            ax.text(x[i] + bar_width, unders[i] + 2, f"{unders[i]:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels([f"Over/Under {mercado}" for mercado in mercados], fontsize=10, rotation=45)
    ax.set_ylabel("Probabilidade (%)", fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend()

    # Melhorar o layout para evitar sobreposição de rótulos
    plt.tight_layout()
    plt.show()
