import os
from PIL import Image
from matplotlib import pyplot as plt

def mostrar_escudo(time, tamanho=(200, 200)):
    """
    Exibe o escudo do time se disponível na pasta 'escudos'.
    Permite redimensionar a imagem para um tamanho específico.
    """
    try:
        # Verifica se a pasta 'escudos' existe, se não, cria a pasta
        if not os.path.exists('escudos'):
            os.makedirs('escudos')
        
        # Lista de formatos de imagem suportados
        formatos_aceitos = ['.png', '.jpg', '.jpeg']
        
        # Tenta encontrar o arquivo da imagem para o time
        escudo_encontrado = False
        for ext in formatos_aceitos:
            caminho_escudo = f'escudos/{time}{ext}'
            if os.path.exists(caminho_escudo):
                escudo_encontrado = True
                break

        # Se escudo não encontrado
        if not escudo_encontrado:
            print(f"⚠️ Escudo do {time} não encontrado ou formato de imagem não suportado.")
            return
        
        # Carrega a imagem
        img = Image.open(caminho_escudo)

        # Redimensiona a imagem, se necessário
        img = img.resize(tamanho)

        # Exibe a imagem com matplotlib
        plt.imshow(img)
        plt.axis('off')  # Desativa os eixos
        plt.title(f"Escudo do {time}", fontsize=14, weight='bold')
        plt.show()

    except Exception as e:
        print(f"❌ Ocorreu um erro ao carregar o escudo do {time}: {e}")
