import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNÇÕES AUXILIARES OTIMIZADAS
# ==========================================
def calcular_centroide(canais, contagens):
    """Calcula o centróide (média ponderada) de um pico."""
    canais = np.array(canais)
    contagens = np.array(contagens)
    
    if np.sum(contagens) == 0:
        return 0
        
    return np.sum(canais * contagens) / np.sum(contagens)

def calcular_desvio_padrao_pico(canais, contagens, centroide):
    """Calcula o desvio padrão (largura/ruído) do pico."""
    canais = np.array(canais)
    contagens = np.array(contagens)
    
    if np.sum(contagens) <= 1:
        return 0
        
    variancia = np.sum(contagens * (canais - centroide)**2) / np.sum(contagens)
    return np.sqrt(variancia)

def ler_espetro(nome_ficheiro, canal_min, canal_max):
    """Lê o ficheiro .asc e extrai canais e contagens."""
    canais = []
    contagens = []
    
    try:
        with open(nome_ficheiro, 'r') as f:
            linhas = f.readlines()
            
        lendo_dados = False
        for linha in linhas:
            if "Chn" in linha and "Counts" in linha:
                lendo_dados = True
                continue
                
            if lendo_dados:
                partes = linha.split(',')
                if len(partes) >= 2:
                    try:
                        canal = int(partes[0].strip())
                        contagem = int(partes[1].strip())
                        
                        if canal_min <= canal <= canal_max:
                            canais.append(canal)
                            contagens.append(contagem)
                    except ValueError:
                        pass
    except FileNotFoundError:
        print(f"ERRO: Não foi possível encontrar '{nome_ficheiro}'.")
                            
    return canais, contagens

# ==========================================
# 2. LEITURA DOS DADOS
# ==========================================
ficheiro_1 = 'populser.asc'
ficheiro_2 = 'ppulser2.asc'
ficheiro_3 = 'ppulser3.asc'

# Valores do dial do pulser
sinal_pulser = np.array([4.0, 5.0, 4.5]) 

# Ler picos Pulser
canais_p1, cont_p1 = ler_espetro(ficheiro_1, 272, 280)
canais_p2, cont_p2 = ler_espetro(ficheiro_2, 856, 864)
canais_p3, cont_p3 = ler_espetro(ficheiro_3, 560, 568)

# Ler picos Polónio
canais_po1, cont_po1 = ler_espetro(ficheiro_1, 300, 380)
canais_po2, cont_po2 = ler_espetro(ficheiro_2, 300, 380)
canais_po3, cont_po3 = ler_espetro(ficheiro_3, 300, 380)

# =======================================
# 3. CÁLCULOS HISTOGRAMA E QUI-QUADRADO
# =======================================
# Centróides Observados (Dados experimentais)
observado = np.array([
    calcular_centroide(canais_p1, cont_p1),
    calcular_centroide(canais_p2, cont_p2),
    calcular_centroide(canais_p3, cont_p3)
])

# Larguras dos picos (Apenas para desenhar as barras de erro no gráfico final)
sigmas_pulser_canais = np.array([
    calcular_desvio_padrao_pico(canais_p1, cont_p1, observado[0]),
    calcular_desvio_padrao_pico(canais_p2, cont_p2, observado[1]),
    calcular_desvio_padrao_pico(canais_p3, cont_p3, observado[2])
])
sigmas_pulser_canais[sigmas_pulser_canais == 0] = 0.5

# Centróides do Polónio
centroides_po = np.array([
    calcular_centroide(canais_po1, cont_po1),
    calcular_centroide(canais_po2, cont_po2),
    calcular_centroide(canais_po3, cont_po3)
])
centroide_po_medio = np.mean(centroides_po)

# --- PASSO A: Ajuste Linear Direto (Sem Pesos) ---
m_p, c_zero = np.polyfit(sinal_pulser, observado, 1)

# Canais Esperados (Modelo Teórico da Reta)
esperado = m_p * sinal_pulser + c_zero

# --- FÓRMULA SOLICITADA: Qui-Quadrado de Pearson ---
chi_quadrado = np.sum(((observado - esperado) ** 2) / esperado)
graus_liberdade = len(sinal_pulser) - 2
chi_quadrado_reduzido = chi_quadrado / graus_liberdade

# Erro Padrão Médio do Sistema (Largura média do ruído eletrónico)
erro_medio_sistema_canais = np.mean(sigmas_pulser_canais)

# --- PASSO B: Calibração em Energia ---
energia_po_real = 5.304
ganho_energia = energia_po_real / (centroide_po_medio - c_zero)
b_energia = - (ganho_energia * c_zero)

# Escalar larguras de canais para Energia (MeV) para o gráfico
sigmas_pulser_energia = sigmas_pulser_canais * ganho_energia

# ==========================================
# 4. APRESENTAÇÃO DOS RESULTADOS
# ==========================================
print("--- 1. ANÁLISE DE INCERTEZA E QUI-QUADRADO ---")
print(f"Canal Zero Eletrónico (C_zero)     : {c_zero:.2f}")
print(f"Largura média dos picos (Ruído)    : ± {erro_medio_sistema_canais:.2f} canais")
print(f"Canais Observados (Dados)          : {observado}")
print(f"Canais Esperados (Modelo)          : {esperado}")
print(f"Qui-Quadrado Total (\u03C7\u00B2)           : {chi_quadrado:.6f}")
print(f"Graus de Liberdade                  : {graus_liberdade}")
print(f"Qui-Quadrado Reduzido (\u03C7\u00B2_nu)      : {chi_quadrado_reduzido:.6f}\n")

print("--- 2. EQUAÇÃO DE CALIBRAÇÃO FINAL ---")
print(f"Ganho: {ganho_energia:.6f} MeV/canal")
print(f"Equação: E(C) = {ganho_energia:.6f} * Canal + ({b_energia:.6f})\n")

# ==========================================
# 5. GRÁFICO FINAL
# ==========================================
energias_pulser_calibradas = ganho_energia * (observado - c_zero)

plt.figure(figsize=(10, 6))

# Reta de calibração
canais_plot = np.array([0, 1024])
energias_plot = ganho_energia * (canais_plot - c_zero)
plt.plot(canais_plot, energias_plot, '--k', label=f'Reta: E = {ganho_energia:.4f}C + ({b_energia:.2f})', alpha=0.7)

# Pontos experimentais do Pulser com as suas larguras associadas
plt.errorbar(observado, energias_pulser_calibradas, 
             yerr=sigmas_pulser_energia, xerr=sigmas_pulser_canais, 
             fmt='bo', markersize=8, capsize=4, 
             label='Picos do Pulser (Largura $\sigma$ do pico)')

# Ponto de referência do Polónio-210
plt.plot(centroide_po_medio, energia_po_real, 'rx', markersize=12, markeredgewidth=3, label=f'Ref: Po-210 ({energia_po_real} MeV)')

# Formatação e Legendas
plt.xlabel('Canal do MCA', fontsize=12)
plt.ylabel('Energia (MeV)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

plt.xlim(0, 1024)
plt.ylim(0, 7)

plt.tight_layout()
plt.show()