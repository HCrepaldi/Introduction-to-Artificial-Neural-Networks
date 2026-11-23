# -*- coding: utf-8 -*-
# =============================================================================
# Introducao as Redes Neurais Artificiais
# Projeto 4 - Redes Neurais Recorrentes (RNNs): LSTM e GRU
# Autor: Henrique Crepaldi - RA 120410
#
# Tema: Previsao de serie temporal de glicemia (previsao t+k). Dado um
#       historico de leituras de glicose, eu quero prever o valor da glicemia
#       k passos a frente.
#
# Fonte dos dados: eu uso o MEU PROPRIO simulador metabolico de Diabetes
# Tipo 1 (DM1), que eu construi originalmente para uma UC de Fisiologia
# Humana 2 (arquivo simulador_dm1.html, incluido neste repositorio). Aqui eu
# porto o motor fisico daquele simulador para Python e o uso para gerar as
# series de glicemia que alimentam as redes recorrentes.
#
# Por que dados do simulador (e nao dados reais): eu convivo com DM1 e tenho
# os relatorios da minha bomba de insulina, mas decidi NAO usa-los para evitar
# expor dados pessoais de saude. O simulador me da series fisiologicamente
# plausiveis, parametrizaveis e que eu entendo e consigo defender 100% na
# banca, pois cada parametro tem significado clinico real (ISF, ICR, bolus,
# basal, glucagon).
# =============================================================================

# Eu configuro o diretorio de cache do matplotlib ANTES de importa-lo, porque
# alguns ambientes tem o HOME somente-leitura. Assim evito o aviso/erro.
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

# Eu rodo em CPU normalmente; se houver GPU disponivel, eu aproveito.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eu crio uma pasta para salvar as figuras do relatorio.
FIG_DIR = "figuras_projeto4"
os.makedirs(FIG_DIR, exist_ok=True)

# Parametros de tempo do simulador (iguais ao HTML original).
TIME_STEP_MINUTES = 5          # cada leitura a cada 5 min (como um CGM real)
HORAS_POR_SIMULACAO = 6        # cada simulacao do motor cobre 6 horas
PASSOS_POR_SIM = int((HORAS_POR_SIMULACAO * 60) / TIME_STEP_MINUTES)  # = 72


# =============================================================================
# 1) MOTOR FISICO DO SIMULADOR DE DM1 (PORTADO DO MEU HTML PARA PYTHON)
# =============================================================================
# Aqui eu reescrevo, passo a passo, a mesma matematica do meu simulador. Mantive
# os mesmos nomes e as mesmas constantes para que a versao Python seja fiel a
# versao JavaScript. O motor soma varios efeitos concorrentes sobre a glicemia:
#   - carboidrato (sobe),
#   - gordura+proteina (sobe, com atraso),
#   - insulina bolus imediata (desce),
#   - insulina estendida / onda dupla (desce, linear),
#   - producao hepatica vs insulina basal (deriva de fundo),
#   - glucagon (defensivo abaixo de 70 mg/dL, ou descontrolado).
def calcular_pontos_glicose(inp):
    """Eu replico calculateGlucosePoints() do meu simulador HTML.

    `inp` e um dicionario com os mesmos campos do simulador. Eu retorno o vetor
    de glicemia (mg/dL) amostrado a cada 5 min ao longo de 6 horas.
    """
    pontos = []

    # Necessidade basal estimada e producao hepatica correspondente. A ideia
    # clinica: o figado produz glicose o tempo todo; a insulina basal compensa.
    necessidade_basal = (inp["weight"] * 0.55 * 0.5) / 24.0
    producao_figado_por_hora = necessidade_basal * inp["isf"]

    # Quanto cada grama de carboidrato sobe a glicemia (depende de ISF e ICR).
    subida_por_carbo = inp["isf"] / inp["icr"]

    # Efeito tardio de gordura+proteina, estimado a partir das calorias.
    kcal_gord_prot = (inp["protein"] * 4.0) + (inp["fat"] * 9.0)
    subida_gord_prot = ((kcal_gord_prot / 100.0) * 10.0 * subida_por_carbo) * 0.6

    # Divisao do bolus em imediato + estendido quando eu uso onda dupla.
    dose_imediata = inp["insulin"]
    dose_estendida = 0.0
    duracao_estendida_min = 0.0
    if inp["dualWaveActive"]:
        frac = inp["dualWaveSplit"] / 100.0
        dose_imediata = inp["insulin"] * frac
        dose_estendida = inp["insulin"] * (1.0 - frac)
        duracao_estendida_min = inp["dualWaveDuration"] * 60.0

    # Queda total de glicose que cada parte da insulina provoca.
    queda_imediata_total = dose_imediata * inp["isf"]
    queda_estendida_total = dose_estendida * inp["isf"]

    glucagon_reativo_acumulado = 0.0

    for i in range(PASSOS_POR_SIM + 1):
        t_min = i * TIME_STEP_MINUTES
        t_horas = t_min / 60.0

        # (4) Fracao da insulina estendida ja absorvida (absorcao linear).
        fracao_estendida = 0.0
        if dose_estendida > 0 and t_min > 15:
            tempo_efetivo = t_min - 15
            fracao_estendida = min(tempo_efetivo / duracao_estendida_min, 1.0)

        # (5) Deriva de fundo: producao do figado menos acao da insulina basal.
        subida_figado = producao_figado_por_hora * t_horas
        queda_basal = (inp["basal"] * inp["isf"]) * t_horas
        deriva_fundo = subida_figado - queda_basal

        # Absorcoes acumuladas (curvas sigmoidais que vao de 0 a 1 no tempo).
        # Carboidrato: absorve rapido (divisor 70, expoente 3).
        c_abs = 1.0 - (1.0 / (1.0 + (t_min / 70.0) ** 3))
        # Gordura/proteina: so comeca depois de 60 min (evito valores negativos).
        if t_min < 60:
            fp_abs = 0.0
        else:
            fp_abs = 1.0 - (1.0 / (1.0 + ((t_min - 60) / 100.0) ** 2))
        # Insulina imediata: absorve um pouco mais devagar que o carboidrato.
        i_imed_abs = 1.0 - (1.0 / (1.0 + (t_min / 80.0) ** 3))

        # Glicose base: eu somo tudo o que sobe e subtraio tudo o que desce.
        glic_base = (
            inp["startGluc"]
            + (inp["carbs"] * subida_por_carbo * c_abs)
            + (subida_gord_prot * fp_abs)
            - (queda_imediata_total * i_imed_abs)
            - (queda_estendida_total * fracao_estendida)
            + deriva_fundo
        )

        # Efeito do glucagon (hormonio que sobe a glicose).
        efeito_glucagon = 0.0
        if inp["glucagonActive"]:
            if inp["glucagonMode"] == "uncontrolled":
                # Modo descontrolado: sobe de forma constante no tempo.
                efeito_glucagon = 30.0 * t_horas
            elif inp["glucagonMode"] == "defensive":
                # Modo defensivo: so age quando a glicose cai abaixo de 70.
                if (glic_base + glucagon_reativo_acumulado) < 70.0:
                    glucagon_reativo_acumulado += 3.0
                efeito_glucagon = glucagon_reativo_acumulado

        glic_final = glic_base + efeito_glucagon
        # Eu nunca permito glicose negativa (nao faz sentido fisiologico).
        pontos.append(max(0.0, glic_final))

    return np.array(pontos, dtype=np.float64)


def inputs_padrao():
    """Eu devolvo os mesmos valores default do meu simulador HTML."""
    return {
        "startGluc": 100.0, "isf": 30.0, "icr": 10.0, "weight": 70.0,
        "basal": 0.8, "carbs": 40.0, "protein": 25.0, "fat": 20.0,
        "insulin": 5.0, "glucagonActive": False, "glucagonMode": "defensive",
        "dualWaveActive": False, "dualWaveSplit": 60.0, "dualWaveDuration": 3.0,
    }


# =============================================================================
# 2) GERACAO DO DATASET: EU ENCADEIO VARIAS SIMULACOES DE 6h
# =============================================================================
# O simulador cobre 6h por rodada. Para ter uma serie longa e variada (como
# dias inteiros de um paciente), eu encadeio muitas simulacoes: cada bloco de
# 6h usa parametros sorteados (refeicao, dose, cenario) e comeca na glicemia em
# que o bloco anterior terminou. Assim a serie fica continua e realista.
def sortear_cenario(rng, glic_inicial):
    """Eu monto um cenario de 6h com refeicao e dose sorteadas.

    Eu vario os parametros clinicos dentro de faixas plausiveis e, de vez em
    quando, ativo cenarios especiais (onda dupla ou glucagon) para enriquecer
    o dataset com situacoes que a rede precisa aprender a prever.
    """
    inp = inputs_padrao()
    inp["startGluc"] = float(np.clip(glic_inicial, 40, 300))

    # Perfil do paciente (eu mantenho relativamente estavel entre blocos).
    inp["isf"] = float(rng.uniform(25, 45))
    inp["icr"] = float(rng.uniform(8, 15))
    inp["basal"] = float(rng.uniform(0.6, 1.1))

    # Refeicao sorteada: as vezes uma refeicao grande, as vezes um lanche leve.
    inp["carbs"] = float(rng.choice([0, 15, 30, 45, 60, 80],
                                    p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1]))
    inp["protein"] = float(rng.uniform(0, 35))
    inp["fat"] = float(rng.uniform(0, 30))

    # Dose de insulina: eu parto da dose "ideal" sugerida pelo bolus calculator
    # do meu simulador e adiciono um erro humano (as vezes a pessoa erra a conta
    # ou chuta a dose). Esse erro e o que gera hiper e hipoglicemias realistas.
    alvo = 100.0
    dose_ideal = max(0.0, inp["carbs"] / inp["icr"]
                     + (inp["startGluc"] - alvo) / inp["isf"])
    erro_humano = rng.normal(0, 0.8)
    inp["insulin"] = float(max(0.0, dose_ideal + erro_humano))

    # Cenario especial 1: onda dupla (refeicao gordurosa) em ~20% dos blocos.
    if rng.random() < 0.2 and inp["carbs"] > 0:
        inp["dualWaveActive"] = True
        inp["dualWaveSplit"] = float(rng.uniform(50, 70))
        inp["dualWaveDuration"] = float(rng.uniform(2, 4))

    # Cenario especial 2: glucagon defensivo (resgate de hipo) sempre ligado,
    # imitando a contrarregulacao fisiologica que protege contra hipoglicemia.
    inp["glucagonActive"] = True
    inp["glucagonMode"] = "defensive"

    return inp


def gerar_serie_do_simulador(n_blocos=80, seed=SEED):
    """Eu encadeio `n_blocos` simulacoes de 6h para formar a serie completa.

    Entre um bloco e outro eu descarto o ponto repetido (o inicio de um bloco
    e o fim do anterior) para nao duplicar amostras. No fim eu adiciono um
    pequeno ruido de sensor, como num CGM real.
    """
    rng = np.random.default_rng(seed)
    serie = []
    glic_atual = 120.0  # glicemia inicial do primeiro bloco

    registros = []  # eu guardo metadados de cada bloco para inspecao
    for b in range(n_blocos):
        inp = sortear_cenario(rng, glic_atual)
        pontos = calcular_pontos_glicose(inp)

        # No primeiro bloco eu uso todos os pontos; nos demais eu descarto o
        # primeiro ponto (que repete o fim do bloco anterior).
        if b == 0:
            serie.extend(pontos.tolist())
        else:
            serie.extend(pontos[1:].tolist())

        glic_atual = pontos[-1]
        registros.append({"bloco": b, "carbs": inp["carbs"],
                          "insulin": round(inp["insulin"], 2),
                          "dual": inp["dualWaveActive"],
                          "glic_fim": round(glic_atual, 1)})

    serie = np.array(serie, dtype=np.float64)

    # Ruido de sensor: pequenas flutuacoes de medicao a cada leitura.
    serie = serie + rng.normal(0, 2.5, size=len(serie))
    serie = np.clip(serie, 40.0, 400.0)

    # Eu monto um DataFrame com o eixo de tempo para os graficos.
    t_min = np.arange(len(serie)) * TIME_STEP_MINUTES
    df = pd.DataFrame({"t_min": t_min, "glicose": serie})
    meta = pd.DataFrame(registros)
    return df, meta


# =============================================================================
# 3) JANELAMENTO: TRANSFORMO A SERIE EM EXEMPLOS (X, y) PARA PREVISAO t+k
# =============================================================================
def criar_janelas(serie, tam_janela, k):
    """Eu transformo a serie 1D em pares (janela de entrada, alvo futuro).

    Para cada instante t eu pego as `tam_janela` leituras ANTERIORES (de
    t-tam_janela+1 ate t) como entrada, e o valor em t+k como alvo. Assim o
    modelo aprende a prever a glicose `k` passos a frente a partir do passado.
    """
    X, y = [], []
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
# 4) MODELOS RECORRENTES: LSTM e GRU
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
# 5) TREINO COM EARLY STOPPING BASEADO NA VALIDACAO
# =============================================================================
def treinar_modelo(modelo, loaders, lr=1e-3, max_epocas=100, paciencia=12,
                   weight_decay=0.0, verbose=False):
    """Eu treino o modelo e uso o conjunto de validacao para parar cedo.

    Eu monitoro a perda de validacao a cada epoca: se ela nao melhora por
    `paciencia` epocas seguidas, eu interrompo o treino (early stopping) e
    restauro os melhores pesos. Isso evita overfitting e me da a curva de
    aprendizagem que eu mostro no relatorio.
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
# 6) AVALIACAO: METRICAS DE REGRESSAO + METRICA CLINICA
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
# 7) GRAFICOS PARA O RELATORIO
# =============================================================================
def plot_serie(df, caminho):
    """Eu ploto um trecho da serie gerada pelo meu simulador."""
    plt.figure(figsize=(11, 4))
    # Eu mostro ~48h para dar para enxergar varios ciclos de refeicao/insulina.
    n = min(len(df), int(48 * 60 / TIME_STEP_MINUTES))
    trecho = df.iloc[:n]
    plt.plot(trecho["t_min"] / 60.0, trecho["glicose"], lw=1.0)
    plt.axhspan(70, 180, color="green", alpha=0.08, label="faixa alvo (70-180)")
    plt.axhline(70, color="red", lw=0.8, ls="--", alpha=0.6)
    plt.axhline(180, color="orange", lw=0.8, ls="--", alpha=0.6)
    plt.xlabel("tempo (horas)")
    plt.ylabel("glicose (mg/dL)")
    plt.title("Serie de glicemia gerada pelo meu simulador de DM1 (48h)")
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
    plt.axhspan(70, 180, color="green", alpha=0.06)
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


def plot_decomposicao_simulador(caminho):
    """Eu ploto UM bloco de 6h do meu simulador decomposto em seus efeitos.

    Esse grafico e so didatico: ele mostra de onde vem a serie de glicemia que
    eu uso para treinar a rede. Eu rodo o motor variando um efeito por vez
    (so carboidrato, so insulina, etc.) para enxergar a contribuicao de cada
    um, alem da curva final (todos os efeitos somados).
    """
    base = inputs_padrao()
    base["carbs"] = 60.0
    base["insulin"] = 6.0
    t = np.arange(PASSOS_POR_SIM + 1) * TIME_STEP_MINUTES / 60.0  # horas

    # Curva final (todos os efeitos juntos).
    curva_final = calcular_pontos_glicose(base)

    # So carboidrato (zero insulina): mostra a subida pos-refeicao.
    so_carbo = dict(base); so_carbo["insulin"] = 0.0
    curva_carbo = calcular_pontos_glicose(so_carbo)

    # So insulina (zero refeicao): mostra a queda provocada pela dose.
    so_insulina = dict(base); so_insulina["carbs"] = 0.0
    so_insulina["protein"] = 0.0; so_insulina["fat"] = 0.0
    curva_insulina = calcular_pontos_glicose(so_insulina)

    plt.figure(figsize=(11, 4.6))
    plt.plot(t, curva_final, color="#2563eb", lw=2.6, label="glicemia final (tudo)")
    plt.plot(t, curva_carbo, color="#16a34a", lw=1.6, ls="--",
             label="so carboidrato (sem insulina)")
    plt.plot(t, curva_insulina, color="#9333ea", lw=1.6, ls="--",
             label="so insulina (sem refeicao)")
    plt.axhspan(70, 180, color="green", alpha=0.07, label="faixa alvo")
    plt.xlabel("tempo (horas)")
    plt.ylabel("glicose (mg/dL)")
    plt.title("Como o meu simulador compoe a glicemia (bloco de 6h)")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(caminho, dpi=130)
    plt.close()


def plot_distribuicao_tir(df, caminho):
    """Eu ploto a distribuicao da glicemia e o Time in Range (TIR).

    TIR (tempo na faixa) e uma metrica clinica central no acompanhamento do
    DM1: a fracao do tempo em que a glicose fica entre 70 e 180 mg/dL. Aqui eu
    mostro o histograma com as tres zonas (hipo, alvo, hiper) e os percentuais.
    """
    g = df["glicose"].values
    frac_hipo = float(np.mean(g < 70)) * 100
    frac_alvo = float(np.mean((g >= 70) & (g <= 180))) * 100
    frac_hiper = float(np.mean(g > 180)) * 100

    plt.figure(figsize=(11, 4.4))
    plt.hist(g, bins=60, color="#64748b", alpha=0.75, edgecolor="white")
    plt.axvspan(40, 70, color="red", alpha=0.10)
    plt.axvspan(70, 180, color="green", alpha=0.10)
    plt.axvspan(180, 400, color="orange", alpha=0.10)
    plt.axvline(70, color="red", ls="--", lw=1)
    plt.axvline(180, color="orange", ls="--", lw=1)
    plt.xlabel("glicose (mg/dL)")
    plt.ylabel("frequencia (leituras)")
    plt.title(f"Distribuicao da glicemia | Hipo={frac_hipo:.1f}%  "
              f"Alvo(TIR)={frac_alvo:.1f}%  Hiper={frac_hiper:.1f}%")
    plt.tight_layout()
    plt.savefig(caminho, dpi=130)
    plt.close()
    return frac_hipo, frac_alvo, frac_hiper


# =============================================================================
# 8) PIPELINE PRINCIPAL
# =============================================================================
def construir_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=64):
    """Eu monto os DataLoaders do PyTorch a partir dos arrays ja normalizados.

    Como cada amostra ja e uma janela independente (X, y), eu POSSO embaralhar
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
    print("Fonte dos dados: meu simulador de DM1 (portado para Python)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # (A) Eu gero a serie encadeando varias simulacoes de 6h e escolho os
    #     parametros do problema de previsao.
    # ------------------------------------------------------------------
    TAM_JANELA = 24        # 24 leituras = 2 horas de historico de entrada
    K = 6                  # prevejo 6 passos = 30 min a frente (t+k)

    df, meta = gerar_serie_do_simulador(n_blocos=80, seed=SEED)
    dias = len(df) * TIME_STEP_MINUTES / 60 / 24
    print(f"Serie gerada: {len(df)} leituras (~{dias:.1f} dias, "
          f"{len(meta)} blocos de 6h).")
    print(f"  glicemia: min={df['glicose'].min():.0f} "
          f"media={df['glicose'].mean():.0f} max={df['glicose'].max():.0f} mg/dL")
    plot_serie(df, os.path.join(FIG_DIR, "01_serie_glicemia.png"))

    # Eu ploto a decomposicao do simulador e a distribuicao/TIR da glicemia.
    plot_decomposicao_simulador(
        os.path.join(FIG_DIR, "05_decomposicao_simulador.png"))
    fh, fa, fhi = plot_distribuicao_tir(
        df, os.path.join(FIG_DIR, "06_distribuicao_tir.png"))
    print(f"  Time in Range: hipo={fh:.1f}% | alvo={fa:.1f}% | hiper={fhi:.1f}%")

    # Eu salvo a serie em CSV para quem quiser inspecionar/reproduzir.
    df.to_csv("serie_glicemia_projeto4.csv", index=False)

    # ------------------------------------------------------------------
    # (B) Eu crio as janelas (X, y) para a previsao t+k.
    # ------------------------------------------------------------------
    serie = df["glicose"].values.astype(np.float32)
    X, y = criar_janelas(serie, TAM_JANELA, K)
    print(f"Janelas criadas: X={X.shape}, y={y.shape} "
          f"(janela={TAM_JANELA} passos, k={K} -> {K*TIME_STEP_MINUTES} min).")

    # ------------------------------------------------------------------
    # (C) Eu divido em treino/val/teste respeitando a ordem temporal.
    # ------------------------------------------------------------------
    (Xtr, ytr), (Xva, yva), (Xte, yte) = dividir_treino_val_teste(X, y)
    print(f"Divisao: treino={len(Xtr)} | val={len(Xva)} | teste={len(Xte)}")

    # ------------------------------------------------------------------
    # (D) Normalizacao SEM VAZAMENTO: eu ajusto o scaler SO no treino e
    #     aplico (transform) em val e teste.
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
    # ------------------------------------------------------------------
    grade = []
    for hidden in [32, 64]:
        for n_layers in [1, 2]:
            for lr in [1e-2, 1e-3]:
                grade.append(dict(hidden_size=hidden, num_layers=n_layers,
                                  lr=lr))

    melhor_cfg = {"LSTM": None, "GRU": None}
    melhor_val = {"LSTM": float("inf"), "GRU": float("inf")}

    for tipo in ["LSTM", "GRU"]:
        print("\n" + "-" * 70)
        print(f"Busca de hiperparametros - {tipo} "
              f"({len(grade)} configuracoes)")
        print("-" * 70)
        for cfg in grade:
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
                max_epocas=60, paciencia=8)
            val = hist["melhor_val"]
            print(f"  hidden={cfg['hidden_size']:>3} "
                  f"layers={cfg['num_layers']} lr={cfg['lr']:<6} "
                  f"-> val_MSE={val:.4f}")
            if val < melhor_val[tipo]:
                melhor_val[tipo] = val
                melhor_cfg[tipo] = cfg

        print(f"  >> melhor {tipo}: {melhor_cfg[tipo]} "
              f"(val_MSE={melhor_val[tipo]:.4f})")

    # ------------------------------------------------------------------
    # (F) Eu re-treino os melhores LSTM e GRU e avalio no TESTE.
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
    # ------------------------------------------------------------------
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

    print("\n" + "=" * 70)
    print(f"Figuras salvas em: {os.path.abspath(FIG_DIR)}/")
    for nome in ["01_serie_glicemia.png", "02_curvas_aprendizagem.png",
                 "03_previsoes_teste.png", "04_dispersao.png",
                 "05_decomposicao_simulador.png", "06_distribuicao_tir.png"]:
        print(f"  - {nome}")
    print(f"Serie salva em: serie_glicemia_projeto4.csv")
    print(f"Tempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
