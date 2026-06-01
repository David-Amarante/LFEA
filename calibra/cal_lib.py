import ROOT
import numpy as np

# Evita o arranque gráfico e desvincula os histogramas da diretoria global do C++
ROOT.gROOT.SetBatch(True)
ROOT.TH1D.AddDirectory(False)

def get_centroid(canais, contagens, idx):
    # Criar o histograma de forma isolada
    h = ROOT.TH1D(f"h_{idx}", "", 1024, 0, 1024)
    for c, n in zip(canais, contagens):
        h.SetBinContent(int(c) + 1, float(n))
    
    centroide = h.GetMean()
    erro = h.GetMeanError()
    return centroide, erro # O Python limpa o 'h' sozinho e em segurança agora

# 1. Obter dados dos ficheiros
arquivos = ['PULSER02.asc', 'PULSER04.ASC', 'PULSER06.ASC', 'PULSER08.ASC', 'PULSER10.ASC', 'PULSER12.ASC']
lista_centroides, lista_erros = [], []

print("--- Calculando Centroides (ROOT TH1D Seguro) ---")
for i, arq in enumerate(arquivos):
    try:
        x, y = np.genfromtxt(arq, skip_header=5, delimiter=",", unpack=True, usecols=(0,1), max_rows=1024)
        c, e = get_centroid(x, y, i)
        lista_centroides.append(c)
        lista_erros.append(e)
        print(f"{arq}: {c:.3f} ± {e:.6f}")
    except Exception as err:
        print(f"Erro no ficheiro {arq}: {err}")

# 2. Configurar Fit Linear (Eixos Invertidos)
x_energias = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float64)
ex = np.full(len(x_energias), 0.001, dtype=np.float64)
y_canais = np.array(lista_centroides, dtype=np.float64)
ey = np.array(lista_erros, dtype=np.float64)

gr = ROOT.TGraphErrors(len(x_energias), x_energias, y_canais, ex, ey)
gr.SetTitle("Calibracao Inversa;Energia (MeV);Canal")
gr.SetMarkerStyle(20)
gr.SetLineColor(ROOT.kBlue)

fit = ROOT.TF1("f", "[0] + [1]*x", float(np.min(x_energias)), float(np.max(x_energias)))
fit.SetLineColor(ROOT.kRed)
gr.Fit(fit, "S Q")

print("\n===== RESULTADOS DO FIT =====")
print(f"p0 = {fit.GetParameter(0):.4f} ± {fit.GetParError(0):.4f}")
print(f"p1 = {fit.GetParameter(1):.4f} ± {fit.GetParError(1):.4f}")
print(f"Chi2/NDF = {fit.GetChisquare()/fit.GetNDF():.4f}\n")

# 3. Gerar Gráfico e Salvar
c = ROOT.TCanvas("c", "", 800, 600)
gr.Draw("APE")
fit.Draw("same")

leg = ROOT.TLegend(0.15, 0.70, 0.45, 0.85)
leg.AddEntry(gr, "Dados", "lp")
leg.AddEntry(fit, "Fit", "l")
leg.Draw()

c.SaveAs("fit_calibracao_inversa.png")
print("Sucesso: Gráfico guardado como 'fit_calibracao_inversa.png'")