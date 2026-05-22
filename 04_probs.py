import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÃO (AJUSTA AQUI)
# ==========================================
caminho_arquivo = "AM57TRY1.asc"
m = 0.002207    # Declive em MeV/canal
b = 4.550775    # Ordenada em MeV

# Define os intervalos dos teus picos [início, fim]
picos_definidos = [
    [455, 520],
    [430, 455],
    [400, 430],
]

# ==========================================
# 2. FUNÇÃO DE LEITURA ROBUSTA
# ==========================================
def ler_ficheiro(caminho):
    canais, contagens = [], []
    with open(caminho, 'r') as f:
        for linha in f:
            partes = linha.replace(',', '.').split()
            if len(partes) >= 2:
                try:
                    canais.append(float(partes[0]))
                    contagens.append(float(partes[1]))
                except ValueError:
                    continue 
    return np.array(canais), np.array(contagens)

# ==========================================
# 3. ANÁLISE E PLOT
# ==========================================
canais, contagens = ler_ficheiro(caminho_arquivo)
total_area = np.sum(contagens)

plt.figure(figsize=(10, 6))
plt.plot(canais, contagens, label='Espectro', color='gray', alpha=0.6, drawstyle='steps-mid')

print(f"{'Pico':<10} | {'Energia (MeV)':<15} | {'Probabilidade (%)':<15}")
print("-" * 45)

for i, [inicio, fim] in enumerate(picos_definidos):
    mask = (canais >= inicio) & (canais <= fim)
    if np.any(mask):
        canal_pico = canais[mask][np.argmax(contagens[mask])]
        area_pico = np.sum(contagens[mask])
        
        # Cálculo direto em MeV
        energia_mev = (m * canal_pico + b)
        prob = (area_pico / total_area) * 100
        
        # Plot do pico
        plt.fill_between(canais[mask], contagens[mask], alpha=0.4, label=f'Pico {i+1} ({energia_mev:.4f} MeV)')
        
        print(f"Pico {i+1:<4} | {energia_mev:<15.4f} | {prob:<15.2f}")

# Estilização
plt.yscale('log')
plt.ylim(bottom=1)
plt.title('Espectro Alpha - Análise de Picos (MeV)')
plt.xlabel('Canais')
plt.ylabel('Contagens (log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()