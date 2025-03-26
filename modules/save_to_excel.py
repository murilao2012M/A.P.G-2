import pandas as pd

# Função para salvar os dados no Excel
def save_to_excel(data, filename="registros_jogos.xlsx"):
    try:
        existing_data = pd.read_excel(filename)
        new_data = pd.concat([existing_data, data], ignore_index=True)
    except FileNotFoundError:
        new_data = data

    new_data.to_excel(filename, index=False)
    print(f"Resultados salvos em {filename}")