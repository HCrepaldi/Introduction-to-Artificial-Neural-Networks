# ==============================================================================
# PROJETO 2 - QUANTIZACAO DE CORES COM APRENDIZADO NAO SUPERVISIONADO
# ------------------------------------------------------------------------------
# Disciplina: Introducao as Redes Neurais Artificiais
# Autor: Henrique Crepaldi (RA: 120410)
#
# Objetivo deste script (com foco DIDATICO):
#   Reduzir drasticamente a paleta de cores de uma imagem (compressao/
#   quantizacao) usando duas redes de aprendizado COMPETITIVO nao
#   supervisionado, e comparar as duas quanto a fidelidade e topologia:
#
#     - SOM  (Self-Organizing Map)  -> biblioteca MiniSom
#     - GNG  (Growing Neural Gas)   -> implementacao PROPRIA (deste arquivo),
#                                       comentada passo a passo (Fritzke, 1995).
#
#   Por que o GNG e implementado do zero? Porque nao ha uma biblioteca de GNG
#   madura e mantida para o Python moderno, e porque o GNG e um algoritmo de
#   poucos passos bem definidos: le-lo direto e a MELHOR forma de entende-lo
#   para a apresentacao (nao ficar refem de uma "caixa-preta").
#
# CAMINHO ADOTADO (Caminho A): treina-se a rede nos pixels de UMA imagem e
#   quantiza-se ESSA MESMA imagem. Isso e COMPRESSAO, nao generalizacao para
#   imagens novas; portanto NAO existe conjunto de teste nem risco de vazamento
#   de dados aqui - essa e a natureza da tarefa de quantizacao.
#
# PIPELINE (conforme o enunciado):
#   1. Extracao ......... carregar imagem, extrair pixels, normalizar RGB p/ [0,1]
#   2. Treinamento ...... SOM (grids 4x4, 8x8, 16x16) e GNG (16, 64, 256 neuronios)
#   3. Hiperparametros .. experimento variando as taxas de aprendizado (SOM e GNG)
#   4. Inferencia ....... para cada pixel, achar o neuronio vencedor (BMU) e
#                         trocar a cor do pixel pela cor dos PESOS desse neuronio
#   5. Reconstrucao ..... remontar a matriz da imagem com as novas cores e exibir
#   6. Avaliacao ........ erro de quantizacao (EQ) e erro topologico (ET)
#
# Como rodar:
#   python -m venv .venv && source .venv/bin/activate
#   pip install numpy pillow matplotlib minisom scipy
#   python quantizacao_cores_project2.py --imagem imagens_treino/gradiente.png
# ==============================================================================

import argparse
import os
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from minisom import MiniSom


# ------------------------------------------------------------------------------
# 1. REPRODUTIBILIDADE
# ------------------------------------------------------------------------------
# Mesma semente do Projeto 1: garante que treino, amostragens e inicializacoes
# aleatorias sejam repetiveis entre execucoes (importante para a banca poder
# reproduzir os numeros do relatorio).
SEED = 42
np.random.seed(SEED)


# ==============================================================================
# 2. EXTRACAO: CARREGAR IMAGEM E NORMALIZAR OS PIXELS
# ==============================================================================
def carregar_pixels(caminho_imagem):
    """Carrega a imagem, extrai todos os pixels e normaliza o RGB para [0, 1].

    Retorna:
      - pixels: array (N, 3) com N = altura*largura pixels, cada um com (R,G,B)
                em ponto flutuante no intervalo [0, 1].
      - forma:  tupla (altura, largura, 3), usada depois para RECONSTRUIR a
                imagem 2D a partir da lista 1D de pixels.

    POR QUE NORMALIZAR PARA [0, 1]?
      Os canais RGB vem como inteiros em [0, 255]. As redes competitivas
      trabalham com DISTANCIAS euclidianas entre pesos e amostras; manter tudo
      em [0, 1] deixa a escala dos tres canais homogenea e estabiliza as taxas
      de aprendizado (um passo de 0.1 significa a mesma coisa em qualquer canal).
    """
    # convert("RGB") descarta um eventual canal alfa (transparencia) e garante
    # exatamente 3 canais por pixel.
    imagem = Image.open(caminho_imagem).convert("RGB")
    matriz = np.asarray(imagem, dtype=np.float32)  # (altura, largura, 3)
    forma = matriz.shape

    # "achatar" a imagem 2D em uma lista de pixels (N, 3). Cada pixel passa a
    # ser uma AMOSTRA 3D independente para o treino (a posicao dele na imagem
    # nao importa para a quantizacao de cor).
    pixels = matriz.reshape(-1, 3) / 255.0
    return pixels, forma


def amostrar_pixels(pixels, n_amostras, semente=SEED):
    """Sorteia um subconjunto de pixels para acelerar o treino.

    POR QUE AMOSTRAR? Uma imagem 4500x4500 tem 20,25 milhoes de pixels. Treinar
    SOM/GNG em todos eles e desnecessario: uma amostra aleatoria de algumas
    dezenas de milhares de pixels ja representa MUITO bem a distribuicao de
    cores. A INFERENCIA (trocar a cor de cada pixel) continua sendo feita em
    TODOS os pixels - a amostragem afeta apenas o TREINO.
    """
    if n_amostras >= len(pixels):
        return pixels
    rng = np.random.default_rng(semente)
    indices = rng.choice(len(pixels), size=n_amostras, replace=False)
    return pixels[indices]


# ==============================================================================
# 3. GROWING NEURAL GAS (GNG) - IMPLEMENTACAO PROPRIA E COMENTADA
# ------------------------------------------------------------------------------
# Referencia: B. Fritzke, "A Growing Neural Gas Network Learns Topologies",
# NIPS 1995. O GNG aprende, ao mesmo tempo, ONDE colocar os neuronios (os
# "prototipos" de cor, que serao a paleta) e QUAIS neuronios sao vizinhos
# (as arestas do grafo, que descrevem a TOPOLOGIA da nuvem de cores).
#
# INTUICAO: comeca com 2 neuronios e vai INSERINDO neuronios nas regioes onde
# o erro acumulado e maior (onde a rede representa mal os dados), ate atingir o
# numero-alvo de neuronios (16, 64 ou 256 = tamanho da paleta desejada).
#
# Diferenca-chave para o SOM: no SOM a topologia e FIXA e definida a priori (o
# grid retangular). No GNG a topologia CRESCE e se ADAPTA aos dados - as arestas
# aparecem e somem conforme a distribuicao real das cores.
# ==============================================================================
class GrowingNeuralGas:
    """GNG para quantizacao vetorial (cada neuronio e um vetor de cor RGB).

    Parametros (hiperparametros do algoritmo, todos com significado explicado):
      n_neuronios_max : tamanho-alvo da paleta (o treino para de crescer ao atingir).
      max_epocas      : quantas passagens completas pela amostra de treino.
      eps_b           : passo de correcao do VENCEDOR (BMU) em direcao ao dado.
      eps_n           : passo de correcao dos VIZINHOS do vencedor (bem menor).
      idade_max       : idade maxima de uma aresta; arestas velhas sao removidas
                        (conexoes que nao sao mais reforcadas "envelhecem").
      lambda_insercao : a cada 'lambda_insercao' amostras processadas, insere-se
                        um novo neuronio (enquanto nao atingir n_neuronios_max).
      alpha           : fator de reducao do erro dos dois neuronios envolvidos
                        na insercao de um novo neuronio.
      beta            : decaimento global de TODOS os erros a cada iteracao
                        (faz o erro recente pesar mais que o antigo).
    """

    def __init__(self, n_neuronios_max=64, max_epocas=10,
                 eps_b=0.05, eps_n=0.006, idade_max=50,
                 lambda_insercao=200, alpha=0.5, beta=0.0005, semente=SEED):
        self.n_neuronios_max = n_neuronios_max
        self.max_epocas = max_epocas
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.idade_max = idade_max
        self.lambda_insercao = lambda_insercao
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(semente)

        # ESTRUTURAS DE DADOS DO GRAFO:
        # - pesos: lista de vetores 3D (as cores-prototipo). E a "posicao" de
        #   cada neuronio no espaco RGB.
        # - erro: erro acumulado de cada neuronio (quanto ele "erra" ao ser o
        #   vencedor). Guia ONDE inserir novos neuronios.
        # - arestas: dicionario {(i, j): idade} com i < j. A existencia da chave
        #   diz que i e j sao vizinhos; o valor e a idade da conexao.
        self.pesos = []
        self.erro = []
        self.arestas = {}

    # --- utilitarios de aresta (mantem sempre i < j para nao duplicar) --------
    @staticmethod
    def _chave(i, j):
        return (i, j) if i < j else (j, i)

    def _conectar(self, i, j):
        """Cria (ou zera a idade de) a aresta entre i e j."""
        self.arestas[self._chave(i, j)] = 0

    def _vizinhos(self, i):
        """Lista os indices de neuronios diretamente conectados a i."""
        viz = []
        for (a, b) in self.arestas:
            if a == i:
                viz.append(b)
            elif b == i:
                viz.append(a)
        return viz

    # --- passo 0: inicializacao -----------------------------------------------
    def _inicializar(self, dados):
        """Comeca com DOIS neuronios em posicoes distintas tiradas dos dados.

        Sorteamos a partir das cores UNICAS da amostra para garantir que os dois
        primeiros neuronios nao nasçam exatamente com a mesma cor (o que causaria
        empate de distancia em imagens com grandes areas de fundo chapado).
        """
        cores_unicas = np.unique(dados, axis=0)
        fonte = cores_unicas if len(cores_unicas) >= 2 else dados
        idx = self.rng.choice(len(fonte), size=2, replace=False)
        self.pesos = [fonte[idx[0]].astype(np.float64).copy(),
                      fonte[idx[1]].astype(np.float64).copy()]
        self.erro = [0.0, 0.0]
        self.arestas = {}
        self._conectar(0, 1)

    # --- passo 1: achar os dois neuronios mais proximos de uma amostra --------
    def _dois_mais_proximos(self, x):
        """Retorna (s1, s2, dist_s1): indices do vencedor (BMU) e do 2o colocado,
        e a distancia AO QUADRADO do vencedor (usada para acumular o erro).
        """
        P = np.asarray(self.pesos)              # (K, 3)
        d2 = np.sum((P - x) ** 2, axis=1)       # (K,) distancias^2
        s1 = int(np.argmin(d2))
        dist_s1 = float(d2[s1])
        d2[s1] = np.inf                         # "esconde" o vencedor
        s2 = int(np.argmin(d2))                 # 2o mais proximo
        return s1, s2, dist_s1

    # --- passo principal: uma apresentacao de amostra ao GNG ------------------
    def _passo(self, x):
        """Executa UMA iteracao do GNG para uma amostra x (um pixel).

        Segue exatamente a sequencia de Fritzke (1995):
        """
        # (1) achar vencedor s1 e segundo colocado s2.
        s1, s2, dist_s1 = self._dois_mais_proximos(x)

        # (2) envelhecer TODAS as arestas que saem do vencedor s1 (o tempo passa
        #     para as conexoes dele; as que nao forem reforcadas vao "morrer").
        vizinhos_s1 = self._vizinhos(s1)
        for j in vizinhos_s1:
            self.arestas[self._chave(s1, j)] += 1

        # (3) acumular o erro do vencedor com a distancia^2 ate a amostra.
        #     Regioes mal representadas acumulam erro alto -> receberao neuronios.
        self.erro[s1] += dist_s1

        # (4) MOVER o vencedor um pouco em direcao a amostra (passo eps_b), e
        #     mover os VIZINHOS dele um pouco menos (passo eps_n). E o "puxao"
        #     competitivo: o prototipo vencedor se aproxima da cor observada.
        self.pesos[s1] += self.eps_b * (x - self.pesos[s1])
        for j in vizinhos_s1:
            self.pesos[j] += self.eps_n * (x - self.pesos[j])

        # (5) conectar s1 e s2 (ou zerar a idade se ja eram vizinhos). Observar
        #     os dois mais proximos juntos e evidencia de que sao vizinhos na
        #     topologia dos dados - essa e a "aprendizagem de topologia".
        self._conectar(s1, s2)

        # (6) remover arestas velhas demais (idade > idade_max) e, em seguida,
        #     remover neuronios que ficaram SEM nenhuma aresta (isolados).
        self._remover_arestas_velhas()

        # (7) decaimento global do erro (todo mundo esquece um pouco do passado).
        fator_decaimento = 1.0 - self.beta
        for i in range(len(self.erro)):
            self.erro[i] *= fator_decaimento

    def _remover_arestas_velhas(self):
        """Remove arestas com idade acima do limite e neuronios isolados."""
        velhas = [k for k, idade in self.arestas.items() if idade > self.idade_max]
        if not velhas:
            return
        for k in velhas:
            del self.arestas[k]

        # 6b. tirar neuronios que ficaram sem nenhuma conexao.
        conectados = set()
        for (a, b) in self.arestas:
            conectados.add(a)
            conectados.add(b)
        if len(conectados) == len(self.pesos):
            return  # ninguem isolado

        # manter apenas neuronios conectados; se sobrarem <2, nao remove nada
        manter = sorted(conectados)
        if len(manter) < 2:
            return
        self._reindexar(manter)

    def _reindexar(self, manter):
        """Reconstroi pesos/erro/arestas mantendo apenas os indices em 'manter'."""
        mapa = {antigo: novo for novo, antigo in enumerate(manter)}
        self.pesos = [self.pesos[i] for i in manter]
        self.erro = [self.erro[i] for i in manter]
        novas = {}
        for (a, b), idade in self.arestas.items():
            if a in mapa and b in mapa:
                novas[self._chave(mapa[a], mapa[b])] = idade
        self.arestas = novas

    # --- passo de crescimento: inserir um novo neuronio -----------------------
    def _inserir_neuronio(self):
        """Insere um neuronio na regiao de MAIOR erro (onde a rede representa
        pior os dados). Este e o mecanismo de CRESCIMENTO do GNG.
        """
        q = int(np.argmax(self.erro))
        viz = self._vizinhos(q)
        if not viz:
            return

        # f: vizinho de q com maior erro.
        f = max(viz, key=lambda j: self.erro[j])

        # r: novo neuronio no ponto medio das cores de q e f.
        novo_peso = 0.5 * (self.pesos[q] + self.pesos[f])
        self.pesos.append(novo_peso)
        r = len(self.pesos) - 1

        # rearranjo das arestas: remove q-f, cria q-r e r-f.
        chave_qf = self._chave(q, f)
        if chave_qf in self.arestas:
            del self.arestas[chave_qf]
        self._conectar(q, r)
        self._conectar(r, f)

        # ajuste dos erros: q e f "dividem" o erro (multiplicados por alpha) e o
        # novo neuronio r herda o erro reduzido de q.
        self.erro[q] *= self.alpha
        self.erro[f] *= self.alpha
        self.erro.append(self.erro[q])

    # --- laco de treino -------------------------------------------------------
    def treinar(self, dados):
        """Treina o GNG na amostra de pixels 'dados' (N, 3)."""
        dados = np.asarray(dados, dtype=np.float64)
        self._inicializar(dados)

        contador = 0
        for _ in range(self.max_epocas):
            ordem = self.rng.permutation(len(dados))
            for idx in ordem:
                self._passo(dados[idx])
                contador += 1

                # crescimento controlado: so insere se ainda nao atingiu o alvo.
                if (contador % self.lambda_insercao == 0
                        and len(self.pesos) < self.n_neuronios_max):
                    self._inserir_neuronio()
        return self

    # --- interface de quantizacao ---------------------------------------------
    def paleta(self):
        """Retorna a paleta aprendida (K, 3) em [0, 1] - os pesos dos neuronios."""
        return np.clip(np.asarray(self.pesos), 0.0, 1.0)

    def quantizar(self, pixels):
        """INFERENCIA: para cada pixel, acha o BMU e devolve o INDICE dele."""
        return _bmu_indices(np.asarray(pixels), self.paleta())


# ==============================================================================
# 4. SOM (Self-Organizing Map) VIA MiniSom - EMBRULHO DIDATICO
# ------------------------------------------------------------------------------
# O SOM tambem e aprendizado competitivo, mas com topologia FIXA: os neuronios
# vivem num grid 2D (ex.: 8x8 = 64 neuronios) e o treino atualiza o vencedor E
# seus vizinhos NO GRID (funcao de vizinhanca gaussiana que encolhe com o tempo).
# ==============================================================================
def treinar_som(dados, lado, num_iteracoes=None,
                sigma=None, learning_rate=0.5, semente=SEED):
    """Treina um SOM quadrado 'lado x lado' nos pixels 'dados' (N, 3).

    Parametros:
      lado            : lado do grid (4, 8 ou 16 -> 16, 64 ou 256 neuronios).
      num_iteracoes   : numero de amostras apresentadas. Se None, escala com o
                        numero de neuronios do grid para nao subtreinar K=256.
      sigma           : raio inicial da vizinhanca no grid (default = metade do
                        lado, para comecar cooperativo e ir especializando).
      learning_rate   : taxa de aprendizado inicial (decai ao longo do treino).
    """
    k = lado * lado
    if num_iteracoes is None:
        # Escala o numero de iteracoes com o tamanho do grid (minimo de 15.000
        # e ate ~38.400 para 16x16), garantindo convergencia justa em K=256.
        num_iteracoes = max(15000, k * 150)

    if sigma is None:
        sigma = max(1.0, lado / 2.0)

    som = MiniSom(
        x=lado, y=lado, input_len=3,
        sigma=sigma, learning_rate=learning_rate,
        neighborhood_function="gaussian",
        random_seed=semente,
    )
    som.random_weights_init(dados)
    som.train_random(dados, num_iteracoes, verbose=False)
    return som


def paleta_som(som):
    """Extrai a paleta (K, 3) de um SOM treinado: um vetor de cor por neuronio
    do grid, achatando o grid 2D (lado x lado) em uma lista (lado*lado, 3)."""
    pesos = som.get_weights()             # (lado, lado, 3)
    return np.clip(pesos.reshape(-1, 3), 0.0, 1.0)


# ==============================================================================
# 5. INFERENCIA GENERICA (BMU) E RECONSTRUCAO DA IMAGEM
# ==============================================================================
def _bmu_indices(pixels, paleta, bloco=100_000):
    """Para cada pixel, retorna o indice da cor mais proxima na paleta (o BMU).

    POR QUE EM BLOCOS? Calcular a distancia de ate ~20 milhoes de pixels contra
    256 cores de uma vez estouraria a memoria RAM. Processar em blocos de 100 mil
    pixels mantem o uso de RAM baixo e totalmente vetorizado.
    """
    pixels = np.asarray(pixels, dtype=np.float32)
    paleta = np.asarray(paleta, dtype=np.float32)
    n = len(pixels)
    indices = np.empty(n, dtype=np.int32)
    for ini in range(0, n, bloco):
        fim = min(ini + bloco, n)
        p = pixels[ini:fim][:, None, :]                     # (b, 1, 3)
        d2 = np.sum((p - paleta[None, :, :]) ** 2, axis=2)  # (b, K)
        indices[ini:fim] = np.argmin(d2, axis=1)
    return indices


def reconstruir_imagem(indices, paleta, forma):
    """RECONSTRUCAO: monta a imagem quantizada substituindo cada pixel pela cor
    do seu BMU e devolve um array uint8 (altura, largura, 3) pronto para salvar.
    """
    novas_cores = paleta[indices]                     # (N, 3) em [0, 1]
    img = (novas_cores.reshape(forma) * 255.0)        # volta para [0, 255]
    return np.clip(img, 0, 255).astype(np.uint8)


# ==============================================================================
# 6. METRICAS: ERRO DE QUANTIZACAO E ERRO TOPOLOGICO (VETORIZADOS)
# ==============================================================================
def erro_quantizacao(pixels, paleta, indices=None, bloco=200_000):
    """Erro de quantizacao = distancia MEDIA (euclidiana) entre cada pixel e a
    cor do seu neuronio vencedor. Calculado em blocos para economizar memoria
    em imagens de altissima resolucao (ex.: 4500x4500).
    """
    pixels = np.asarray(pixels, dtype=np.float32)
    paleta = np.asarray(paleta, dtype=np.float32)
    if indices is None:
        indices = _bmu_indices(pixels, paleta)

    soma_dist = 0.0
    n = len(pixels)
    for ini in range(0, n, bloco):
        fim = min(ini + bloco, n)
        diffs = pixels[ini:fim] - paleta[indices[ini:fim]]
        soma_dist += float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
    return soma_dist / n


def erro_topologico_som(pixels, som, bloco=50_000):
    """Erro topologico do SOM = fracao de amostras cujo 1o e 2o BMU NAO sao
    vizinhos no grid (distancia de Chebyshev > 1).

    Como a relacao de vizinhanca e simetrica (b1 vizinho de b2 <=> b2 vizinho
    de b1), basta pegar os 2 menores indices via argpartition de forma
    totalmente vetorizada!
    """
    pesos = som.get_weights()                 # (lado_x, lado_y, 3)
    _, lado_y, _ = pesos.shape
    grade = pesos.reshape(-1, 3).astype(np.float32)
    pixels = np.asarray(pixels, dtype=np.float32)

    violacoes = 0
    for ini in range(0, len(pixels), bloco):
        p = pixels[ini:ini + bloco][:, None, :]            # (b, 1, 3)
        d2 = np.sum((p - grade[None, :, :]) ** 2, axis=2)  # (b, K)
        dois = np.argpartition(d2, 1, axis=1)[:, :2]       # (b, 2)
        x1, y1 = np.divmod(dois[:, 0], lado_y)
        x2, y2 = np.divmod(dois[:, 1], lado_y)
        dist_chebyshev = np.maximum(np.abs(x1 - x2), np.abs(y1 - y2))
        violacoes += int(np.sum(dist_chebyshev > 1))
    return violacoes / len(pixels)


def erro_topologico_gng(pixels, gng, bloco=50_000):
    """Erro topologico do GNG = fracao de amostras cujo 1o e 2o BMU NAO estao
    ligados por uma ARESTA do grafo aprendido.

    Constroi uma matriz booleana de adjacencia (K x K) para verificar todas as
    amostras do bloco instantaneamente via indexacao matricial NumPy.
    """
    paleta = gng.paleta().astype(np.float32)
    k = len(paleta)
    pixels = np.asarray(pixels, dtype=np.float32)

    # Matriz de adjacencia simetrica (K, K) das arestas aprendidas
    adj = np.zeros((k, k), dtype=bool)
    for (a, b) in gng.arestas:
        adj[a, b] = True
        adj[b, a] = True

    violacoes = 0
    for ini in range(0, len(pixels), bloco):
        p = pixels[ini:ini + bloco][:, None, :]
        d2 = np.sum((p - paleta[None, :, :]) ** 2, axis=2)
        dois = np.argpartition(d2, 1, axis=1)[:, :2]
        conectados = adj[dois[:, 0], dois[:, 1]]
        violacoes += int(np.sum(~conectados))
    return violacoes / len(pixels)


# ==============================================================================
# 7. VISUALIZACAO: ORIGINAL vs QUANTIZADA + PALETA + GRAFICOS DE METRICAS
# ==============================================================================
def salvar_comparacao(caminho_saida, original_uint8, quantizada_uint8,
                      paleta, titulo, n_cores):
    """Salva uma figura com a imagem original, a quantizada e a paleta de cores
    extraida (os prototipos aprendidos), no estilo do slide do enunciado."""
    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(original_uint8)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(quantizada_uint8)
    ax2.set_title(f"Quantizada ({titulo}, K={n_cores})")
    ax2.axis("off")

    # faixa de paleta: uma linha de "swatches" com as cores aprendidas.
    ax3 = fig.add_subplot(2, 1, 2)
    faixa = np.clip(paleta, 0, 1).reshape(1, -1, 3)
    ax3.imshow(faixa, aspect="auto")
    ax3.set_title("Paleta de cores extraida (centroides / pesos dos neuronios)")
    ax3.set_yticks([])
    ax3.set_xlabel(f"{len(paleta)} cores")

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=130)
    plt.close(fig)


def salvar_grafico_comparativo(caminho_saida, resultados):
    """Gera um grafico de barras comparando SOM x GNG em EQ e ET, por K."""
    ks = [16, 64, 256]
    eq_som = {k: 0.0 for k in ks}
    eq_gng = {k: 0.0 for k in ks}
    et_som = {k: 0.0 for k in ks}
    et_gng = {k: 0.0 for k in ks}
    for rede, _cfg, k, eq, et, _dt in resultados:
        alvo_eq = eq_som if rede == "SOM" else eq_gng
        alvo_et = et_som if rede == "SOM" else et_gng
        k_alvo = min(ks, key=lambda kk: abs(kk - k))
        alvo_eq[k_alvo] = eq
        alvo_et[k_alvo] = et

    x = np.arange(len(ks))
    largura = 0.35
    fig, (axq, axt) = plt.subplots(1, 2, figsize=(12, 4.5))

    axq.bar(x - largura / 2, [eq_som[k] for k in ks], largura, label="SOM")
    axq.bar(x + largura / 2, [eq_gng[k] for k in ks], largura, label="GNG")
    axq.set_title("Erro de Quantizacao (menor = melhor)")
    axq.set_xlabel("Numero de cores (K)")
    axq.set_ylabel("EQ (distancia media)")
    axq.set_xticks(x)
    axq.set_xticklabels(ks)
    axq.legend()
    axq.grid(True, axis="y", ls="--", alpha=0.5)

    axt.bar(x - largura / 2, [et_som[k] for k in ks], largura, label="SOM")
    axt.bar(x + largura / 2, [et_gng[k] for k in ks], largura, label="GNG")
    axt.set_title("Erro Topologico (menor = melhor)")
    axt.set_xlabel("Numero de cores (K)")
    axt.set_ylabel("ET (fracao de violacoes)")
    axt.set_xticks(x)
    axt.set_xticklabels(ks)
    axt.legend()
    axt.grid(True, axis="y", ls="--", alpha=0.5)

    fig.suptitle("SOM vs GNG - comparacao de metricas por numero de cores")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=130)
    plt.close(fig)


# ==============================================================================
# 8. EXPERIMENTO DE HIPERPARAMETROS (TAXAS DE APRENDIZADO - SLIDE 3)
# ==============================================================================
def executar_experimento_taxas(dados_treino, pixels_avaliacao, pasta_saida):
    """Cumpre o requisito do Slide 3: 'Considerar ajustes nos hiperparametros
    (taxas) e avaliar as redes geradas usando o erro de quantizacao e topologico'.

    Fixamos a escala intermediaria (K=64: SOM 8x8 e GNG 64) e testamos tres
    regimes de taxa de aprendizado:
      - Baixa : passos muito timidos (rede demora a abrir e cobrir o espaco RGB)
      - Media : configuracao equilibrada (padrao adotado no script)
      - Alta  : passos agressivos (os prototipos 'saltam' demais a cada amostra,
                causando instabilidade e dobrando/rasgando a topologia).
    """
    print("\n" + "=" * 74)
    print("EXPERIMENTO DE HIPERPARAMETROS (TAXAS DE APRENDIZADO) - Escala K=64")
    print("=" * 74)

    taxas_som = [
        ("Baixa (0.05)", 0.05),
        ("Media (0.50)", 0.50),
        ("Alta (1.50)",  1.50),
    ]
    taxas_gng = [
        ("Baixa (0.005 / 0.0005)", 0.005, 0.0005),
        ("Media (0.050 / 0.0060)", 0.050, 0.0060),
        ("Alta (0.300 / 0.0500)",  0.300, 0.0500),
    ]

    res_som = []
    print("[SOM 8x8 - Variando learning_rate]")
    for rotulo, lr in taxas_som:
        t0 = time.time()
        som = treinar_som(dados_treino, lado=8, learning_rate=lr)
        pal = paleta_som(som)
        eq = erro_quantizacao(pixels_avaliacao, pal)
        et = erro_topologico_som(dados_treino, som)
        dt = time.time() - t0
        print(f"  Taxa {rotulo:<22} -> EQ={eq:.4f} | ET={et:.4f} | tempo={dt:.1f}s")
        res_som.append((rotulo, eq, et))

    res_gng = []
    print("\n[GNG 64 - Variando (eps_b / eps_n)]")
    for rotulo, eb, en in taxas_gng:
        t0 = time.time()
        gng = GrowingNeuralGas(n_neuronios_max=64, max_epocas=10, eps_b=eb, eps_n=en)
        gng.treinar(dados_treino)
        pal = gng.paleta()
        eq = erro_quantizacao(pixels_avaliacao, pal)
        et = erro_topologico_gng(dados_treino, gng)
        dt = time.time() - t0
        print(f"  Taxa {rotulo:<22} -> EQ={eq:.4f} | ET={et:.4f} | tempo={dt:.1f}s")
        res_gng.append((rotulo, eq, et))

    # Salva grafico do experimento de taxas para incluir nos slides/relatorio
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    rotulos_s = [r[0] for r in res_som]
    ax1.bar(rotulos_s, [r[1] for r in res_som], color="#1f77b4", width=0.5)
    ax1.set_title("SOM 8x8: Erro de Quantizacao por Taxa")
    ax1.set_ylabel("EQ (menor = melhor)")
    ax1.grid(True, axis="y", ls="--", alpha=0.5)

    ax2.bar(rotulos_s, [r[2] for r in res_som], color="#1f77b4", width=0.5)
    ax2.set_title("SOM 8x8: Erro Topologico por Taxa")
    ax2.set_ylabel("ET (menor = melhor)")
    ax2.grid(True, axis="y", ls="--", alpha=0.5)

    rotulos_g = [r[0] for r in res_gng]
    ax3.bar(rotulos_g, [r[1] for r in res_gng], color="#ff7f0e", width=0.5)
    ax3.set_title("GNG 64: Erro de Quantizacao por Taxa")
    ax3.set_ylabel("EQ (menor = melhor)")
    ax3.grid(True, axis="y", ls="--", alpha=0.5)

    ax4.bar(rotulos_g, [r[2] for r in res_gng], color="#ff7f0e", width=0.5)
    ax4.set_title("GNG 64: Erro Topologico por Taxa")
    ax4.set_ylabel("ET (menor = melhor)")
    ax4.grid(True, axis="y", ls="--", alpha=0.5)

    fig.suptitle("Impacto das Taxas de Aprendizado em K=64 (SOM vs GNG)")
    fig.tight_layout()
    caminho_fig = os.path.join(pasta_saida, "experimento_taxas_k64.png")
    fig.savefig(caminho_fig, dpi=130)
    plt.close(fig)
    print(f"[OK] Grafico do experimento de taxas salvo: {caminho_fig}")


# ==============================================================================
# 9. EXECUCAO PRINCIPAL
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Quantizacao de cores com SOM e GNG (Projeto 2)."
    )
    parser.add_argument("--imagem", required=True,
                        help="Caminho da imagem de entrada (PNG/JPG).")
    parser.add_argument("--amostras", type=int, default=20000,
                        help="Nº de pixels amostrados para TREINO (default 20000).")
    parser.add_argument("--epocas-gng", type=int, default=10,
                        help="Nº de epocas de treino do GNG (default 10).")
    parser.add_argument("--saida", default="resultados",
                        help="Pasta onde salvar as figuras de resultado.")
    parser.add_argument("--pular-experimento-taxas", action="store_true",
                        help="Se passado, pula o estudo de variacao de taxas.")
    args = parser.parse_args()

    os.makedirs(args.saida, exist_ok=True)

    # ----- 1. EXTRACAO --------------------------------------------------------
    print(f"[INFO] Carregando imagem: {args.imagem}")
    pixels, forma = carregar_pixels(args.imagem)
    original_uint8 = (pixels.reshape(forma) * 255).astype(np.uint8)
    print(f"       Dimensoes: {forma[1]}x{forma[0]} px ({len(pixels):,} pixels)")

    dados_treino = amostrar_pixels(pixels, args.amostras)
    print(f"       Amostra de treino: {len(dados_treino):,} pixels "
          f"({len(np.unique(dados_treino, axis=0)):,} cores unicas na amostra)\n")

    resultados = []  # linhas da tabela final

    # ----- 2/4/5/6. SOM: grids 4x4, 8x8, 16x16 --------------------------------
    print("=" * 74)
    print("SOM (Self-Organizing Map) - MiniSom")
    print("=" * 74)
    for lado in (4, 8, 16):
        k = lado * lado
        t0 = time.time()
        som = treinar_som(dados_treino, lado=lado)
        pal = paleta_som(som)
        idx = _bmu_indices(pixels, pal)
        eq = erro_quantizacao(pixels, pal, idx)
        et = erro_topologico_som(dados_treino, som)
        quant = reconstruir_imagem(idx, pal, forma)
        nome = os.path.join(args.saida, f"som_{lado}x{lado}.png")
        salvar_comparacao(nome, original_uint8, quant, pal,
                          titulo=f"SOM {lado}x{lado}", n_cores=k)
        dt = time.time() - t0
        print(f"  Grid {lado}x{lado:<2} (K={k:<3})  "
              f"EQ={eq:.4f}  ET={et:.4f}  tempo={dt:.1f}s  -> {nome}")
        resultados.append(("SOM", f"{lado}x{lado}", k, eq, et, dt))

    # ----- 2/4/5/6. GNG: 16, 64, 256 neuronios --------------------------------
    print("\n" + "=" * 74)
    print("GNG (Growing Neural Gas) - implementacao propria")
    print("=" * 74)
    for k in (16, 64, 256):
        t0 = time.time()
        gng = GrowingNeuralGas(n_neuronios_max=k, max_epocas=args.epocas_gng)
        gng.treinar(dados_treino)
        pal = gng.paleta()
        idx = gng.quantizar(pixels)
        eq = erro_quantizacao(pixels, pal, idx)
        et = erro_topologico_gng(dados_treino, gng)
        quant = reconstruir_imagem(idx, pal, forma)
        nome = os.path.join(args.saida, f"gng_{k}.png")
        salvar_comparacao(nome, original_uint8, quant, pal,
                          titulo=f"GNG {k}", n_cores=len(pal))
        dt = time.time() - t0
        print(f"  Neuronios={k:<3} (obtidos={len(pal):<3})  "
              f"EQ={eq:.4f}  ET={et:.4f}  tempo={dt:.1f}s  -> {nome}")
        resultados.append(("GNG", str(k), len(pal), eq, et, dt))

    # ----- TABELA FINAL -------------------------------------------------------
    print("\n" + "=" * 74)
    print("RESUMO  (EQ = erro de quantizacao | ET = erro topologico)")
    print("=" * 74)
    print(f"{'Rede':<5} {'Config':<8} {'K':<5} {'EQ':<10} {'ET':<10} {'tempo(s)':<9}")
    print("-" * 74)
    for rede, cfg, k, eq, et, dt in resultados:
        print(f"{rede:<5} {cfg:<8} {k:<5} {eq:<10.4f} {et:<10.4f} {dt:<9.1f}")
    print("=" * 74)

    # grafico de barras comparativo (SOM x GNG, EQ e ET por K).
    nome_cmp = os.path.join(args.saida, "comparativo_metricas.png")
    salvar_grafico_comparativo(nome_cmp, resultados)
    print(f"[OK] Grafico comparativo salvo: {nome_cmp}")

    # ----- 3. EXPERIMENTO DE HIPERPARAMETROS (TAXAS) --------------------------
    if not args.pular_experimento_taxas:
        executar_experimento_taxas(dados_treino, pixels, args.saida)

    print(f"\n[OK] Processamento concluido! Figuras salvas em: {args.saida}/")


if __name__ == "__main__":
    main()