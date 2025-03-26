import os
import pandas as pd

def carregar_dados_excel(arquivos, tipo_arquivo='excel'):
    """
    Carrega dados de partidas a partir de arquivos, calcula as médias de gols de mandante e visitante.

    Parâmetros:
        arquivos (list): Lista de arquivos para carregar.
        tipo_arquivo (str): Tipo de arquivo ('excel', 'csv', etc.).

    Retorna:
        dict: Contendo as médias de gols de mandante, visitante, e os dados completos.
    """
    dfs = []
    for arquivo in arquivos:
        if os.path.exists(arquivo):
            try:
                if tipo_arquivo == 'excel':
                    df = pd.read_excel(arquivo)
                elif tipo_arquivo == 'csv':
                    df = pd.read_csv(arquivo)
                else:
                    raise ValueError(f"Tipo de arquivo '{tipo_arquivo}' não suportado.")

                # Verificar se as colunas esperadas estão presentes
                required_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"Coluna '{col}' não encontrada no arquivo {arquivo}.")

                # Adicionar ao DataFrame completo
                dfs.append(df)

            except Exception as e:
                print(f"Erro ao carregar o arquivo {arquivo}: {e}")

    if dfs:
        df_completo = pd.concat(dfs, ignore_index=True)

        # Garantir que os dados não contenham valores nulos nas colunas relevantes
        df_completo.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'], inplace=True)

        # Calcular as médias de gols
        media_gols_mandante = df_completo.groupby('HomeTeam')['FTHG'].mean()
        media_gols_visitante = df_completo.groupby('AwayTeam')['FTAG'].mean()

        # Calcular também as médias de gols sofridos e diferença de gols
        media_gols_sofridos_mandante = df_completo.groupby('HomeTeam')['FTAG'].mean()
        media_gols_sofridos_visitante = df_completo.groupby('AwayTeam')['FTHG'].mean()

        return {
            'media_gols_mandante': media_gols_mandante,
            'media_gols_visitante': media_gols_visitante,
            'media_gols_sofridos_mandante': media_gols_sofridos_mandante,
            'media_gols_sofridos_visitante': media_gols_sofridos_visitante,
            'dados_partidas': df_completo
        }
    else:
        raise FileNotFoundError("Nenhum arquivo Excel ou CSV encontrado.")
