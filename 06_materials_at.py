import numpy as np

def ler_counts(ficheiro):
    """
    Lê um ficheiro ASC do GMX-PC e extrai apenas as contagens
    das 10 medições individuais.
    """
    counts = []
    with open(ficheiro, "r") as f:
        for linha in f:
            partes = linha.strip().split()
            if partes and partes[0].replace(",", "").isdigit():
                try:
                    counts.append(int(partes[1]))
                except:
                    pass
    return np.array(counts)


# ---------------------------------------------------------
# CONFIGURAÇÃO DO UTILIZADOR
# ---------------------------------------------------------

arquivos = [
    "calmstat.asc",
    "calpbstat.asc",
    "calstat.asc",
    "calcustat.asc",
    "tlalstat.asc",
    "tlcustat.asc",
    "tlpbstat.asc",
    "tlmstat.asc"
]

arquivo_background = "f2stat.asc"   # <-- tu colocas o nome aqui


# ---------------------------------------------------------
# PROCESSAMENTO DO BACKGROUND
# ---------------------------------------------------------

bg_counts = ler_counts(arquivo_background)
bg_media_10s = np.mean(bg_counts)
bg_std_10s = np.std(bg_counts, ddof=1)

# Normalizar para 20 s
bg_media_20s = 2.0 * bg_media_10s
bg_std_20s = 2.0 * bg_std_10s


# ---------------------------------------------------------
# PROCESSAMENTO DOS ARQUIVOS PRINCIPAIS
# ---------------------------------------------------------

resultados = []

for arq in arquivos:
    counts = ler_counts(arq)
    media = np.mean(counts)
    std = np.std(counts, ddof=1)

    # Subtrair background normalizado
    media_corr = media - bg_media_20s

    # Propagação de erro
    std_corr = np.sqrt(std**2 + bg_std_20s**2)

    # Rate e erro do rate
    rate = media_corr / 20.0
    rate_err = std_corr / 20.0

    resultados.append((arq, media, std, media_corr, std_corr, rate, rate_err))


# ---------------------------------------------------------
# TABELA FINAL
# ---------------------------------------------------------

print("Tabela Final (com correção de background e erros):")
print(f"{'Arquivo':20} {'Média ± σ':20} {'Corrigida ± σ':25} {'Rate ± σ (s^-1)':20}")

for arq, media, std, media_corr, std_corr, rate, rate_err in resultados:
    print(f"{arq:20} "
          f"{media:8.2f} ± {std:6.2f}   "
          f"{media_corr:8.2f} ± {std_corr:6.2f}   "
          f"{rate:6.3f} ± {rate_err:6.3f}")
