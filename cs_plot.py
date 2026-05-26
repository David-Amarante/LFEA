import pandas as pd
import matplotlib.pyplot as plt
import io

def plotar_espectro(caminho_arquivo):
    # 1. Ler o arquivo e encontrar a linha onde começam os dados
    with open(caminho_arquivo, 'r') as f:
        linhas = f.readlines()
    
    # Encontrar a linha que contém o cabeçalho "Chn, Counts, ROI"
    start_row = 0
    for i, linha in enumerate(linhas):
        if "Chn" in linha and "Counts" in linha:
            start_row = i + 1
            break
    
    # 2. Carregar os dados (ignorando o cabeçalho complexo)
    # Ajustamos para ler apenas a parte numérica (canais e contagens)
    df = pd.read_csv(caminho_arquivo, skiprows=start_row, usecols=[0, 1], names=['Canal', 'Contagens'])
    
    # 3. Plotagem
    plt.figure(figsize=(10, 6))
    plt.plot(df['Canal'], df['Contagens'], drawstyle='steps-mid', color='blue', linewidth=1)
    
    plt.title(f"Espectro: {caminho_arquivo}")
    plt.xlabel("Canal")
    plt.ylabel("Contagens")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Salvar e mostrar
    nome_imagem = caminho_arquivo.replace('.asc', '.png')
    plt.savefig(nome_imagem)
    plt.show()
    print(f"Gráfico salvo como: {nome_imagem}")



plotar_espectro('cpulser2.asc')