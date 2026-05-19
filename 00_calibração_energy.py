import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNÇÕES AUXILIARES
# ==========================================
def calcular_centroide(canais, contagens):
    """Calcula o centróide (média ponderada) de um pico."""
    canais = np.array(canais)
    contagens = np.array(contagens)
    
    if np.sum(contagens) == 0:
        return 0
        
    return np.sum(canais * contagens) / np.sum(contagens)

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

# ==========================================
# 3. CÁLCULOS E ESTATÍSTICA DE RESÍDUOS
# ==========================================
centroides_pulser = np.array([
    calcular_centroide(canais_p1, cont_p1),
    calcular_centroide(canais_p2, cont_p2),
    calcular_centroide(canais_p3, cont_p3)
])

centroides_po = np.array([
    calcular_centroide(canais_po1, cont_po1),
    calcular_centroide(canais_po2, cont_po2),
    calcular_centroide(canais_po3, cont_po3)
])
centroide_po_medio = np.mean(centroides_po)

# --- PASSO A: Regressão Simples do Pulser ---
m_p, c_zero = np.polyfit(sinal_pulser, centroides_pulser, 1)

# Calcular o Erro do Sistema a partir dos Resíduos
canais_esperados = m_p * sinal_pulser + c_zero
residuos = centroides_pulser - canais_esperados

graus_liberdade = len(sinal_pulser) - 2
soma_residuos_quadrado = np.sum(residuos**2)
sigma_sistema_canais = np.sqrt(soma_residuos_quadrado / graus_liberdade)

# Cálculo do R^2
ss_tot = np.sum((centroides_pulser - np.mean(centroides_pulser))**2)
r_quadrado = 1 - (soma_residuos_quadrado / ss_tot)

# --- PASSO B: Calibração em Energia ---
energia_po_real = 5.304
ganho_energia = energia_po_real / (centroide_po_medio - c_zero)
b_energia = - (ganho_energia * c_zero)

# Propagar a incerteza dos canais para a Energia (MeV)
sigma_sistema_energia = sigma_sistema_canais * ganho_energia

# ==========================================
# 4. APRESENTAÇÃO DOS RESULTADOS
# ==========================================
print("--- 1. ANÁLISE DE ERRO DO SISTEMA ---")
print(f"Canal Zero Eletrónico (C_zero) : {c_zero:.2f}")
print(f"Erro Padrão Global do Sist.(\u03C3) : \u00B1 {sigma_sistema_canais:.2f} canais")
print(f"Coeficiente R\u00B2                : {r_quadrado:.6f}\n")

print("--- 2. EQUAÇÃO DE CALIBRAÇÃO FINAL ---")
print(f"Ganho: {ganho_energia:.6f} MeV/canal")
print(f"Equação: E(C) = {ganho_energia:.6f} * Canal + ({b_energia:.6f})\n")

# ==========================================
# 5. GRÁFICO FINAL
# ==========================================
energias_pulser_calibradas = ganho_energia * (centroides_pulser - c_zero)

plt.figure(figsize=(10, 6))

# Reta de calibração
canais_plot = np.array([0, 1024])
energias_plot = ganho_energia * (canais_plot - c_zero)
plt.plot(canais_plot, energias_plot, '--k', label=f'Reta: E = {ganho_energia:.4f}C + ({b_energia:.2f})', alpha=0.7)

# Pontos do Pulser (agora com barras de erro que refletem a incerteza real do sistema)
plt.errorbar(centroides_pulser, energias_pulser_calibradas, 
             yerr=sigma_sistema_energia, xerr=sigma_sistema_canais, 
             fmt='bo', markersize=8, capsize=4, 
             label=f'Picos do Pulser (\u00B1 {sigma_sistema_canais:.1f} canais)')

# Ponto do Polónio (Marcador 'x' vermelho e espesso)
plt.plot(centroide_po_medio, energia_po_real, 'rx', markersize=12, markeredgewidth=3, label=f'Ref: Po-210 ({energia_po_real} MeV)')

# Formatação e Legendas (Sem Título)
plt.xlabel('Canal do MCA', fontsize=12)
plt.ylabel('Energia (MeV)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

# Limites rígidos exigidos
plt.xlim(0, 1024)
plt.ylim(0, 7)

plt.tight_layout()
plt.show()