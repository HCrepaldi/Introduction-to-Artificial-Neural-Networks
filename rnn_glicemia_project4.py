# -*- coding: utf-8 -*-
# =============================================================================
# Introducao as Redes Neurais Artificiais
# Projeto 4 - Redes Neurais Recorrentes (RNNs): LSTM e GRU
# Autor: Henrique Crepaldi - RA 120410
#
# Tema: Previsao de serie temporal de glicemia (CGM - Continuous Glucose
#       Monitoring). Dado um historico de leituras de glicose, eu quero
#       prever o valor da glicemia k passos a frente (previsao t+k).
#
# Observacao importante sobre os dados: eu NAO uso dados reais da minha bomba
# de insulina para evitar qualquer questao de privacidade / dados pessoais de
# saude. No lugar, eu gero uma serie sintetica fisiologicamente plausivel
# (ritmo circadiano, picos pos-refeicao, efeito da insulina e ruido de sensor).
# Essa e uma pratica padrao e defensavel em ML aplicado a saude.
# =============================================================================

# Eu configuro o diretorio de cache do matplotlib ANTES de importa-lo, porque
# o ambiente onde rodo tem o HOME somente-leitura. Assim evito o aviso/erro.
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem interface grafica: eu so salvo os PNGs
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------------------------------------------------------
# Reprodutibilidade: eu fixo a mesma SEED em todas as fontes de aleatoriedade
# para que meus resultados sejam reproduziveis (exigencia que eu mantenho em
# todos os projetos da disciplina).
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Eu rodo em CPU neste ambiente; se houver GPU disponivel, eu aproveito.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eu crio uma pasta para salvar as figuras do relatorio.
FIG_DIR = "figuras_projeto4"
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# 1) GERACAO DA SERIE TEMPORAL SINTETICA DE GLICEMIA (CGM)
# =============================================================================
def gerar_serie_glicemia(n_dias=60, passo_min=5, seed=SEED):
    """Eu gero uma serie de glicemia simulando um sensor CGM.

    A ideia e produzir um sinal realista o suficiente para que o problema de
    previsao t+k seja interessante, mas sem usar dados reais. Eu modelo:
      - uma linha de base (nivel basal de glicose);
      - um ritmo circadiano (a glicose oscila ao longo do dia);
      - picos pos-refeicao (cafe, almoco, jantar) que sobem rapido e decaem;
      - o efeito da insulina (puxa a glicose para baixo depois das refeicoes);
      - ruido de sensor (pequenas flutuacoes de medicao).

    Parametros:
      n_dias   : quantos dias de dados eu simulo.
      passo_min: intervalo entre leituras, em minutos (CGM real ~5 min).
      seed     : semente local para reproduzir exatamente a mesma serie.

    Retorno:
      df: DataFrame com colunas de tempo e a leitura de glicose (mg/dL).
    """
    rng = np.random.default_rng(seed)

    # Numero de leituras por dia e no total.
    leituras_por_dia = int(24 * 60 / passo_min)
    n = n_dias * leituras_por_dia

    # Vetor de tempo em minutos desde o inicio e a "hora do dia" (0-24h).
    t_min = np.arange(n) * passo_min
    hora_do_dia = (t_min / 60.0) % 24.0

    # (a) Nivel basal de glicose (mg/dL). Um valor saudavel de jejum.
    basal = 100.0

    # (b) Ritmo circadiano: eu uso uma senoide suave ao longo de 24h. A glicose
    # tende a ser um pouco mais alta de manha (fenomeno do alvorecer).
    circadiano = 12.0 * np.sin(2 * np.pi * (hora_do_dia - 6) / 24.0)

    # (c) Picos pos-refeicao: para cada dia eu disparo tres refeicoes em horarios
    # proximos de 7h, 12h e 19h, com variacao aleatoria de horario e tamanho.
    refeicoes = np.zeros(n)
    insulina = np.zeros(n)
    for dia in range(n_dias):
        for hora_base, tam_base in [(7.0, 55.0), (12.0, 70.0), (19.0, 60.0)]:
            # Eu sorteio pequenas variacoes de horario e intensidade por dia.
            hora = hora_base + rng.normal(0, 0.4)
            tamanho = max(10.0, tam_base + rng.normal(0, 15.0))

            # Indice da leitura em que a refeicao comeca.
            idx_inicio = dia * leituras_por_dia + int(hora * 60 / passo_min)
            if idx_inicio >= n:
                continue

            # (c.1) Absorcao dos carboidratos: eu modelo o pico como uma curva
            # que sobe e desce (funcao gama-like) a partir do inicio da refeicao.
            dur = int(180 / passo_min)  # ~3h de efeito
            k_idx = np.arange(dur)
            forma = (k_idx / 6.0) * np.exp(-k_idx / 9.0)  # sobe e decai
            forma = forma / (forma.max() + 1e-9)
            fim = min(idx_inicio + dur, n)
            refeicoes[idx_inicio:fim] += tamanho * forma[: fim - idx_inicio]

            # (c.2) Efeito da insulina (bolus): comeca um pouco depois da
            # refeicao e puxa a glicose para baixo, tambem como uma curva suave.
            atraso = int(20 / passo_min)
            idx_ins = idx_inicio + atraso
            if idx_ins < n:
                forma_ins = (k_idx / 8.0) * np.exp(-k_idx / 12.0)
                forma_ins = forma_ins / (forma_ins.max() + 1e-9)
                fim_i = min(idx_ins + dur, n)
                insulina[idx_ins:fim_i] -= (
                    0.75 * tamanho * forma_ins[: fim_i - idx_ins]
                )

    # (d) Ruido de sensor: pequenas flutuacoes de medicao a cada leitura.
    ruido = rng.normal(0, 3.0, size=n)

    # Eu componho o sinal final somando todos os efeitos.
    glicose = basal + circadiano + refeicoes + insulina + ruido

    # Eu limito a faixa a valores fisiologicamente plausiveis (mg/dL).
    glicose = np.clip(glicose, 40.0, 400.0)

    df = pd.DataFrame(
        {
            "t_min": t_min,
            "hora_do_dia": hora_do_dia,
            "glicose": glicose,
        }
    )
    return df


# =============================================================================
# 2) JANELAMENTO: TRANSFORMO A SERIE EM EXEMPLOS (X, y) PARA PREVISAO t+k
# =============================================================================
def criar_janelas(serie, tam_janela, k):
    """Eu transformo a serie 1D em pares (janela de entrada, alvo futuro).

    Para cada instante t eu pego as `tam_janela` leituras ANTERIORES (de
    t-tam_janela+1 ate t) como entrada, e o valor em t+k como alvo. Assim o
    modelo aprende a prever a glicose `k` passos a frente a partir do passado.

    Exemplo: tam_janela=24 (2h, se o passo e 5 min) e k=6 (30 min a frente).

    Retorno:
      X: array (n_amostras, tam_janela) com as janelas de entrada.
      y: array (n_amostras,) com o alvo em t+k.
    """
    X, y = [], []
    # Eu paro cedo o suficiente para que o alvo t+k ainda exista na serie.
    for t in range(tam_janela - 1, len(serie) - k):
        janela = serie[t - tam_janela + 1 : t + 1]
        alvo = serie[t + k]
        X.append(janela)
        y.append(alvo)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def dividir_treino_val_teste(X, y, frac_treino=0.70, frac_val=0.15):
    """Eu divido os dados RESPEITANDO A ORDEM TEMPORAL.

    Em serie temporal eu NAO posso embaralhar antes de dividir, senao eu teria
    vazamento de dados (o modelo veria o futuro durante o treino). Por isso eu
    corto a serie em blocos contiguos: treino (passado), validacao (meio) e
    teste (futuro). O conjunto de validacao e o que eu uso para ajustar os
    hiperparametros, e o teste eu so toco no final para reportar o desempenho.
    """
    n = len(X)
    i_treino = int(n * frac_treino)
    i_val = int(n * (frac_treino + frac_val))

    X_tr, y_tr = X[:i_treino], y[:i_treino]
    X_va, y_va = X[i_treino:i_val], y[i_treino:i_val]
    X_te, y_te = X[i_val:], y[i_val:]
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


# =============================================================================
# 3) MODELOS RECORRENTES: LSTM e GRU
# =============================================================================
class RedeRecorrente(nn.Module):
    """Eu defino uma unica classe que serve tanto para LSTM quanto para GRU.

    A diferenca entre as duas e so o tipo de celula recorrente; o resto da
    arquitetura (entrada, empilhamento de camadas, dropout e a camada densa
    final que gera a previsao) eu mantenho igual para a comparacao ser justa.
    """

    def __init__(self, tipo="LSTM", n_features=1, hidden_size=64,
                 num_layers=1, dropout=0.0):
        super().__init__()
        self.tipo = tipo.upper()

        # Eu escolho a celula recorrente conforme o tipo pedido.
        celula = nn.LSTM if self.tipo == "LSTM" else nn.GRU

        # batch_first=True: eu trabalho com tensores (batch, tempo, features),
        # que e a convencao que acho mais intuitiva de ler.
        self.rnn = celula(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # O dropout do PyTorch so atua ENTRE camadas empilhadas; por isso
            # eu so aplico quando tenho mais de uma camada.
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Camada densa final: pego o ultimo estado oculto e projeto em 1 valor
        # (a glicose prevista em t+k). E uma regressao, entao a saida e linear.
        self.cabeca = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x tem forma (batch, tempo, features).
        saida, _ = self.rnn(x)
        # Eu uso o estado do ULTIMO passo de tempo como resumo da janela.
        ultimo = saida[:, -1, :]
        return self.cabeca(ultimo).squeeze(-1)


# =============================================================================
# 4) TREINO COM EARLY STOPPING BASEADO NA VALIDACAO
# =============================================================================
def treinar_modelo(modelo, loaders, lr=1e-3, max_epocas=100, paciencia=12,
                   weight_decay=0.0, verbose=False):
    """Eu treino o modelo e uso o conjunto de validacao para parar cedo.

    Eu monitoro a perda de validacao a cada epoca: se ela nao melhora por
    `paciencia` epocas seguidas, eu interrompo o treino (early stopping) e
    restauro os melhores pesos. Isso evita overfitting e me da a curva de
    aprendizagem que eu mostro no relatorio.

    Retorno:
      hist: dicionario com as listas de perda de treino e de validacao.
    """
    dl_treino, dl_val = loaders
    modelo = modelo.to(DEVICE)

    # Funcao de perda de regressao (erro quadratico medio) e otimizador Adam.
    criterio = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    melhor_val = float("inf")
    melhor_estado = None
    epocas_sem_melhora = 0
    hist = {"treino": [], "val": []}

    for epoca in range(max_epocas):
        # ---- fase de treino ----
        modelo.train()
        perda_treino = 0.0
        for xb, yb in dl_treino:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            otimizador.zero_grad()
            pred = modelo(xb)
            perda = criterio(pred, yb)
            perda.backward()
            otimizador.step()
            perda_treino += perda.item() * len(xb)
        perda_treino /= len(dl_treino.dataset)

        # ---- fase de validacao (sem calcular gradiente) ----
        modelo.eval()
        perda_val = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = modelo(xb)
                perda_val += criterio(pred, yb).item() * len(xb)
        perda_val /= len(dl_val.dataset)

        hist["treino"].append(perda_treino)
        hist["val"].append(perda_val)

        if verbose:
            print(f"  epoca {epoca+1:03d} | treino={perda_treino:.4f} "
                  f"| val={perda_val:.4f}")

        # ---- early stopping: guardo os melhores pesos segundo a validacao ----
        if perda_val < melhor_val - 1e-5:
            melhor_val = perda_val
            melhor_estado = {k: v.detach().clone()
                             for k, v in modelo.state_dict().items()}
            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia:
                if verbose:
                    print(f"  parei cedo na epoca {epoca+1} "
                          f"(melhor val={melhor_val:.4f}).")
                break

    # Eu restauro os melhores pesos antes de devolver o modelo.
    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)

    hist["melhor_val"] = melhor_val
    return hist


# =============================================================================
# 5) AVALIACAO: METRICAS DE REGRESSAO + METRICA CLINICA
# =============================================================================
def prever(modelo, dl):
    """Eu rodo o modelo em um DataLoader e devolvo (previsoes, alvos)."""
    modelo.eval()
    preds, alvos = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            preds.append(modelo(xb).cpu().numpy())
            alvos.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(alvos)


def metricas_clinicas(y_true, y_pred):
    """Eu calculo uma metrica clinica simples inspirada nas zonas de erro.

    Em glicemia, nem todo erro tem o mesmo peso clinico. Aqui eu reporto a
    fracao de previsoes "clinicamente aceitaveis", que eu defino como estar
    dentro de 15 mg/dL (para valores baixos) ou de 15% (para valores altos)
    do valor real. E uma aproximacao do criterio 15/15 usado na pratica.
    """
    erro_abs = np.abs(y_pred - y_true)
    limite = np.where(y_true <= 100.0, 15.0, 0.15 * y_true)
    aceitaveis = erro_abs <= limite
    return float(np.mean(aceitaveis)) * 100.0


def avaliar(modelo, dl, scaler_y, nome=""):
    """Eu avalio o modelo em mg/dL (desnormalizando) e reporto as metricas."""
    pred_norm, alvo_norm = prever(modelo, dl)

    # Eu desfaco a normalizacao para interpretar os erros em mg/dL reais.
    pred = scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
    alvo = scaler_y.inverse_transform(alvo_norm.reshape(-1, 1)).ravel()

    rmse = math.sqrt(mean_squared_error(alvo, pred))
    mae = mean_absolute_error(alvo, pred)
    r2 = r2_score(alvo, pred)
    clin = metricas_clinicas(alvo, pred)

    print(f"  [{nome}] RMSE={rmse:.2f} mg/dL | MAE={mae:.2f} mg/dL "
          f"| R2={r2:.3f} | aceitaveis(15/15)={clin:.1f}%")
    return {"rmse": rmse, "mae": mae, "r2": r2, "clin": clin,
            "pred": pred, "alvo": alvo}


# =============================================================================
# 6) GRAFICOS PARA O RELATORIO
# =============================================================================
def plot_serie(df, caminho):
    """Eu ploto um trecho da serie gerada para ilustrar o dado de entrada."""
    plt.figure(figsize=(11, 4))
    # Eu mostro so os primeiros 3 dias para dar para enxergar os padroes.
    leituras_3dias = int(3 * 24 * 60 / 5)
    trecho = df.iloc[:leituras_3dias]
    plt.plot(trecho["t_min"] / 60.0, trecho["glicose"], lw=1.0)
    plt.axhspan(70, 180, color="green", alpha=0.08, label="faixa alvo (70-180)")
    plt.xlabel("tempo (horas)")
    plt.ylabel("glicose (mg/dL)")
    plt.title("Serie sintetica de glicemia (primeiros 3 dias)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(caminho, dpi=130)
    plt.close()


def plot_curvas_aprendizagem(hists, caminho):
    """Eu ploto as curvas de aprendizagem (treino e validacao) de cada modelo."""
    plt.figure(figsize=(11, 4.5))
    for nome, hist in hists.items():
        plt.plot(hist["treino"], label=f"{nome} - treino", lw=1.6)
        plt.plot(hist["val"], "--", label=f"{nome} - validacao", lw=1.6)
    plt.xlabel("epoca")
    plt.ylabel("perda (MSE, dados normalizados)")
    plt.title("Curvas de aprendizagem: LSTM vs GRU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho, dpi=130)
    plt.close()


def plot_previsoes(resultados, caminho, n_pontos=300):
    """Eu comparo previsao x valor real no conjunto de teste."""
    plt.figure(figsize=(11, 4.5))
    primeiro = next(iter(resultados.values()))
    plt.plot(primeiro["alvo"][:n_pontos], color="black", lw=2.0,
             label="valor real")
    for nome, r in resultados.items():
        plt.plot(r["pred"][:n_pontos], lw=1.3, alpha=0.9,
                 label=f"previsao {nome}")
    plt.xlabel("passo de tempo (teste)")
    plt.ylabel("glicose (mg/dL)")
    plt.title("Previsao t+k no conjunto de teste")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho, dpi=130)
    plt.close()


def plot_dispersao(resultados, caminho):
    """Eu ploto previsto x real (dispersao) para ver o vies de cada modelo."""
    n = len(resultados)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, (nome, r) in zip(axes, resultados.items()):
        ax.scatter(r["alvo"], r["pred"], s=6, alpha=0.3)
        lo = min(r["alvo"].min(), r["pred"].min())
        hi = max(r["alvo"].max(), r["pred"].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal (y=x)")
        ax.set_xlabel("glicose real (mg/dL)")
        ax.set_ylabel("glicose prevista (mg/dL)")
        ax.set_title(f"{nome}  (R2={r['r2']:.3f})")
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(caminho, dpi=130)
    plt.close()


# =============================================================================
# 7) PIPELINE PRINCIPAL
# =============================================================================
def construir_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=64):
    """Eu monto os DataLoaders do PyTorch a partir dos arrays ja normalizados.

    Eu NAO embaralho treino aqui para nao quebrar a nocao temporal? Na verdade,
    como cada amostra ja e uma janela independente (X, y), eu POSSO embaralhar
    as amostras de treino com seguranca: o vazamento foi evitado la na divisao
    temporal dos blocos. Embaralhar ajuda o gradiente. Val e teste eu mantenho
    em ordem para os graficos ficarem legiveis.
    """
    def _tds(X, y):
        # Eu adiciono a dimensao de feature: (amostras, tempo) -> (..., 1).
        Xt = torch.tensor(X).unsqueeze(-1)
        yt = torch.tensor(y)
        return TensorDataset(Xt, yt)

    dl_tr = DataLoader(_tds(Xtr, ytr), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(_tds(Xva, yva), batch_size=batch_size, shuffle=False)
    dl_te = DataLoader(_tds(Xte, yte), batch_size=batch_size, shuffle=False)
    return dl_tr, dl_va, dl_te


def main():
    t0 = time.time()
    print("=" * 70)
    print("Projeto 4 - RNNs (LSTM e GRU) | Previsao t+k de glicemia")
    print("=" * 70)

    # ------------------------------------------------------------------
    # (A) Eu gero a serie e escolho os parametros do problema.
    # ------------------------------------------------------------------
    PASSO_MIN = 5          # cada leitura a cada 5 min (como um CGM real)
    TAM_JANELA = 24        # 24 leituras = 2 horas de historico de entrada
    K = 6                  # prevejo 6 passos = 30 min a frente (t+k)

    df = gerar_serie_glicemia(n_dias=60, passo_min=PASSO_MIN, seed=SEED)
    print(f"Serie gerada: {len(df)} leituras "
          f"({len(df)*PASSO_MIN/60/24:.0f} dias).")
    plot_serie(df, os.path.join(FIG_DIR, "01_serie_glicemia.png"))

    # ------------------------------------------------------------------
    # (B) Eu crio as janelas (X, y) para a previsao t+k.
    # ------------------------------------------------------------------
    serie = df["glicose"].values.astype(np.float32)
    X, y = criar_janelas(serie, TAM_JANELA, K)
    print(f"Janelas criadas: X={X.shape}, y={y.shape} "
          f"(janela={TAM_JANELA} passos, k={K} -> {K*PASSO_MIN} min).")

    # ------------------------------------------------------------------
    # (C) Eu divido em treino/val/teste respeitando a ordem temporal.
    # ------------------------------------------------------------------
    (Xtr, ytr), (Xva, yva), (Xte, yte) = dividir_treino_val_teste(X, y)
    print(f"Divisao: treino={len(Xtr)} | val={len(Xva)} | teste={len(Xte)}")

    # ------------------------------------------------------------------
    # (D) Normalizacao SEM VAZAMENTO: eu ajusto o scaler SO no treino e
    #     aplico (transform) em val e teste. Nunca uso estatisticas do
    #     futuro (val/teste) para normalizar o passado (treino).
    # ------------------------------------------------------------------
    scaler_X = StandardScaler().fit(Xtr)
    scaler_y = StandardScaler().fit(ytr.reshape(-1, 1))

    Xtr_n = scaler_X.transform(Xtr).astype(np.float32)
    Xva_n = scaler_X.transform(Xva).astype(np.float32)
    Xte_n = scaler_X.transform(Xte).astype(np.float32)
    ytr_n = scaler_y.transform(ytr.reshape(-1, 1)).ravel().astype(np.float32)
    yva_n = scaler_y.transform(yva.reshape(-1, 1)).ravel().astype(np.float32)
    yte_n = scaler_y.transform(yte.reshape(-1, 1)).ravel().astype(np.float32)

    # ------------------------------------------------------------------
    # (E) AJUSTE DE HIPERPARAMETROS usando o conjunto de VALIDACAO.
    #     Eu faco uma busca em grade pequena e escolho, para LSTM e GRU
    #     separadamente, a configuracao com menor perda de validacao.
    # ------------------------------------------------------------------
    grade = []
    for hidden in [32, 64]:
        for n_layers in [1, 2]:
            for lr in [1e-2, 1e-3]:
                grade.append(dict(hidden_size=hidden, num_layers=n_layers,
                                  lr=lr))

    resultados_busca = {"LSTM": [], "GRU": []}
    melhor_cfg = {"LSTM": None, "GRU": None}
    melhor_val = {"LSTM": float("inf"), "GRU": float("inf")}

    for tipo in ["LSTM", "GRU"]:
        print("\n" + "-" * 70)
        print(f"Busca de hiperparametros - {tipo} "
              f"({len(grade)} configuracoes)")
        print("-" * 70)
        for cfg in grade:
            # Eu re-semeio antes de cada treino para comparacao justa.
            torch.manual_seed(SEED)
            dl_tr, dl_va, dl_te = construir_loaders(
                Xtr_n, ytr_n, Xva_n, yva_n, Xte_n, yte_n, batch_size=64)
            modelo = RedeRecorrente(
                tipo=tipo, n_features=1,
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=0.2)
            hist = treinar_modelo(
                modelo, (dl_tr, dl_va), lr=cfg["lr"],
                max_epocas=80, paciencia=10)
            val = hist["melhor_val"]
            resultados_busca[tipo].append((cfg, val))
            print(f"  hidden={cfg['hidden_size']:>3} "
                  f"layers={cfg['num_layers']} lr={cfg['lr']:<6} "
                  f"-> val_MSE={val:.4f}")
            if val < melhor_val[tipo]:
                melhor_val[tipo] = val
                melhor_cfg[tipo] = cfg

        print(f"  >> melhor {tipo}: {melhor_cfg[tipo]} "
              f"(val_MSE={melhor_val[tipo]:.4f})")

    # ------------------------------------------------------------------
    # (F) Eu re-treino os melhores LSTM e GRU e avalio no TESTE (que ate
    #     agora nunca foi tocado). Guardo as curvas para o relatorio.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Treino final com a melhor configuracao de cada modelo")
    print("=" * 70)

    hists = {}
    resultados = {}
    for tipo in ["LSTM", "GRU"]:
        cfg = melhor_cfg[tipo]
        torch.manual_seed(SEED)
        dl_tr, dl_va, dl_te = construir_loaders(
            Xtr_n, ytr_n, Xva_n, yva_n, Xte_n, yte_n, batch_size=64)
        modelo = RedeRecorrente(
            tipo=tipo, n_features=1,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=0.2)
        print(f"\n{tipo} com {cfg}:")
        hist = treinar_modelo(modelo, (dl_tr, dl_va), lr=cfg["lr"],
                              max_epocas=120, paciencia=15, verbose=False)
        hists[tipo] = hist

        print("  Desempenho:")
        avaliar(modelo, dl_va, scaler_y, nome=f"{tipo}-validacao")
        resultados[tipo] = avaliar(modelo, dl_te, scaler_y,
                                   nome=f"{tipo}-teste")

    # ------------------------------------------------------------------
    # (G) Baseline ingenuo: prever que a glicose em t+k e igual a de t.
    #     Eu uso isso como referencia para mostrar que a rede aprende algo.
    # ------------------------------------------------------------------
    # A ultima coluna da janela (sem normalizar) e a leitura no instante t.
    base_pred = Xte[:, -1]
    base_rmse = math.sqrt(mean_squared_error(yte, base_pred))
    base_mae = mean_absolute_error(yte, base_pred)
    print("\n" + "-" * 70)
    print(f"Baseline ingenuo (persistencia, y_hat=t): "
          f"RMSE={base_rmse:.2f} mg/dL | MAE={base_mae:.2f} mg/dL")
    print("-" * 70)

    # ------------------------------------------------------------------
    # (H) Eu salvo todos os graficos do relatorio.
    # ------------------------------------------------------------------
    plot_curvas_aprendizagem(
        hists, os.path.join(FIG_DIR, "02_curvas_aprendizagem.png"))
    plot_previsoes(resultados, os.path.join(FIG_DIR, "03_previsoes_teste.png"))
    plot_dispersao(resultados, os.path.join(FIG_DIR, "04_dispersao.png"))

    # ------------------------------------------------------------------
    # (I) Resumo final que eu copio para o relatorio.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESUMO FINAL (conjunto de teste)")
    print("=" * 70)
    print(f"{'modelo':<10}{'RMSE':>10}{'MAE':>10}{'R2':>10}{'15/15(%)':>12}")
    for tipo in ["LSTM", "GRU"]:
        r = resultados[tipo]
        print(f"{tipo:<10}{r['rmse']:>10.2f}{r['mae']:>10.2f}"
              f"{r['r2']:>10.3f}{r['clin']:>12.1f}")
    print(f"{'baseline':<10}{base_rmse:>10.2f}{base_mae:>10.2f}"
          f"{'-':>10}{'-':>12}")

    print(f"\nFiguras salvas em: {os.path.abspath(FIG_DIR)}/")
    print(f"Tempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
