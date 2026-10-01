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
#   3. Inferencia ....... para cada pixel, achar o neuronio vencedor (BMU) e
#                         trocar a cor do pixel pela cor dos PESOS desse neuronio
#   4. Reconstrucao ..... remontar a matriz da imagem com as novas cores e exibir
#   5. Avaliacao ........ erro de quantizacao e erro topologico
#
# Como rodar:
#   python -m venv .venv && source .venv/bin/activate
#   pip install numpy pillow matplotlib minisom scipy
#   python quantizacao_cores_project2.py --imagem imagens_treino/flyer.png
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
    # exatamente 3 canais por pixel - flyers costumam ser PNG com alfa.
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

    POR QUE AMOSTRAR? Uma imagem 1350x1080 tem ~1.46 milhao de pixels. Treinar
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
      n_neuronios_max : tamanho-alvo da paleta (o treino para ao atingir isso).
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

    def __init__(self, n_neuronios_max=64, max_epocas=20,
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
        """Comeca com DOIS neuronios em posicoes aleatorias tiradas dos dados.

        Escolher pontos reais dos dados (e nao vetores aleatorios no cubo RGB)
        ja coloca os prototipos dentro da nuvem de cores desde o inicio.
        """
        idx = self.rng.choice(len(dados), size=2, replace=False)
        self.pesos = [dados[idx[0]].astype(np.float64).copy(),
                      dados[idx[1]].astype(np.float64).copy()]
        self.erro = [0.0, 0.0]
        self.arestas = {}
        self._conectar(0, 1)

    # --- passo 1: achar os dois neuronios mais proximos de uma amostra --------
    def _dois_mais_proximos(self, x):
        """Retorna (s1, s2): indices do vencedor (BMU) e do 2o colocado, e a
        distancia AO QUADRADO do vencedor (usada para acumular o erro).

        Usa distancia euclidiana ao quadrado (nao precisa da raiz para comparar,
        e mais barato). O vencedor s1 e o neuronio cuja cor esta mais proxima da
        cor da amostra x; s2 e o segundo mais proximo.
        """
        P = np.asarray(self.pesos)              # (K, 3)
        d2 = np.sum((P - x) ** 2, axis=1)       # (K,) distancias^2
        s1 = int(np.argmin(d2))
        dist_s1 = d2[s1]
        d2[s1] = np.inf                          # "esconde" o vencedor
        s2 = int(np.argmin(d2))                  # 2o mais proximo
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
        for j in self._vizinhos(s1):
            self.arestas[self._chave(s1, j)] += 1

        # (3) acumular o erro do vencedor com a distancia^2 ate a amostra.
        #     Regioes mal representadas acumulam erro alto -> receberao neuronios.
        self.erro[s1] += dist_s1

        # (4) MOVER o vencedor um pouco em direcao a amostra (passo eps_b), e
        #     mover os VIZINHOS dele um pouco menos (passo eps_n). E o "puxao"
        #     competitivo: o prototipo vencedor se aproxima da cor observada.
        self.pesos[s1] += self.eps_b * (x - self.pesos[s1])
        for j in self._vizinhos(s1):
            self.pesos[j] += self.eps_n * (x - self.pesos[j])

        # (5) conectar s1 e s2 (ou zerar a idade se ja eram vizinhos). Observar
        #     os dois mais proximos juntos e evidencia de que sao vizinhos na
        #     topologia dos dados - essa e a "aprendizagem de topologia".
        self._conectar(s1, s2)

        # (6) remover arestas velhas demais (idade > idade_max) e, em seguida,
        #     remover neuronios que ficaram SEM nenhuma aresta (isolados).
        self._remover_arestas_velhas()

        # (7) decaimento global do erro (todo mundo esquece um pouco do passado).
        for i in range(len(self.erro)):
            self.erro[i] -= self.beta * self.erro[i]

    def _remover_arestas_velhas(self):
        """Remove arestas com idade acima do limite e neuronios isolados."""
        # 6a. tirar arestas velhas.
        velhas = [k for k, idade in self.arestas.items() if idade > self.idade_max]
        for k in velhas:
            del self.arestas[k]

        # 6b. tirar neuronios que ficaram sem nenhuma conexao. Remover um indice
        #     desloca todos os indices seguintes, entao remontamos as estruturas.
        conectados = set()
        for (a, b) in self.arestas:
            conectados.add(a)
            conectados.add(b)
        if len(conectados) == len(self.pesos):
            return  # ninguem isolado

        # manter apenas neuronios conectados; se sobrarem <2, nao remove nada
        # (o GNG precisa de pelo menos 2 neuronios para continuar).
        manter = sorted(conectados)
        if len(manter) < 2:
            return
        self._reindexar(manter)

    def _reindexar(self, manter):
        """Reconstroi pesos/erro/arestas mantendo apenas os indices em 'manter'.

        Cria um mapa "indice antigo -> indice novo" e reescreve tudo. Isso e
        necessario porque nossas arestas guardam indices inteiros posicionais.
        """
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

        Passos classicos:
          q = neuronio com maior erro acumulado.
          f = vizinho de q com maior erro.
          novo neuronio r nasce no PONTO MEDIO entre q e f.
          desfaz a aresta q-f e cria q-r e r-f (r entra "no meio").
          reduz os erros de q e f (fator alpha) e da a r o erro de q.
        """
        # q: maior erro global.
        q = int(np.argmax(self.erro))
        viz = self._vizinhos(q)
        if not viz:
            return  # sem vizinhos nao ha onde inserir (nao deve ocorrer)

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

    # --- laco de treino ---------------------------------------------------------
    def treinar(self, dados):
        """Treina o GNG na amostra de pixels 'dados' (N, 3).

        A cada 'lambda_insercao' amostras processadas, tenta inserir um neuronio
        (ate atingir n_neuronios_max). Percorre os dados por 'max_epocas' vezes,
        embaralhando a ordem a cada epoca.
        """
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

            # se ja atingiu o numero-alvo de neuronios, podemos parar de crescer
            # mas continuamos refinando as posicoes ate acabar as epocas.
        return self

    # --- interface de quantizacao ---------------------------------------------
    def paleta(self):
        """Retorna a paleta aprendida (K, 3) em [0, 1] - os pesos dos neuronios."""
        return np.clip(np.asarray(self.pesos), 0.0, 1.0)

    def quantizar(self, pixels):
        """INFERENCIA: para cada pixel, acha o BMU e devolve o INDICE dele.

        Retorna um array (N,) com o indice do neuronio vencedor de cada pixel.
        A cor final do pixel sera paleta()[indice].
        """
        return _bmu_indices(np.asarray(pixels), self.paleta())


# ==============================================================================
# 4. SOM (Self-Organizing Map) VIA MiniSom - EMBRULHO DIDATICO
# ------------------------------------------------------------------------------
# O SOM tambem e aprendizado competitivo, mas com topologia FIXA: os neuronios
# vivem num grid 2D (ex.: 8x8 = 64 neuronios) e o treino atualiza o vencedor E
# seus vizinhos NO GRID (funcao de vizinhanca gaussiana que encolhe com o tempo).
# Usamos a MiniSom porque e madura, estavel e de codigo pequeno/legivel.
# ==============================================================================
def treinar_som(dados, lado, num_iteracoes=5000,
                sigma=None, learning_rate=0.5, semente=SEED):
    """Treina um SOM quadrado 'lado x lado' nos pixels 'dados' (N, 3).

    Parametros:
      lado            : lado do grid (4, 8 ou 16 -> 16, 64 ou 256 neuronios).
      num_iteracoes   : numero de amostras apresentadas (treino aleatorio).
      sigma           : raio inicial da vizinhanca no grid (default = metade do
                        lado, para comecar cooperativo e ir especializando).
      learning_rate   : taxa de aprendizado inicial (decai ao longo do treino).

    A entrada tem 3 dimensoes (R, G, B), entao input_len=3.
    """
    if sigma is None:
        # raio inicial ~ metade do grid: no comeco muitos vizinhos se movem
        # juntos (organizacao global); com o tempo sigma decai e o ajuste vira
        # local (refinamento fino de cada regiao de cor).
        sigma = max(1.0, lado / 2.0)

    som = MiniSom(
        x=lado, y=lado, input_len=3,
        sigma=sigma, learning_rate=learning_rate,
        neighborhood_function="gaussian",
        random_seed=semente,
    )
    # inicializa os pesos amostrando os proprios dados (comeca dentro da nuvem
    # de cores, como fizemos no GNG).
    som.random_weights_init(dados)
    # train_random apresenta amostras sorteadas; decaimento padrao de lr e sigma.
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

    POR QUE EM BLOCOS? Calcular a distancia de ~1.46M de pixels contra ate 256
    cores de uma vez cria uma matriz gigante (N x K). Processar em blocos de
    100 mil pixels mantem o uso de memoria baixo sem perder velocidade.

    Formula da distancia^2 (sem raiz, pois so precisamos comparar):
      ||p - c||^2 para cada pixel p e cada cor c da paleta; escolhemos o argmin.
    """
    pixels = np.asarray(pixels, dtype=np.float32)
    paleta = np.asarray(paleta, dtype=np.float32)
    n = len(pixels)
    indices = np.empty(n, dtype=np.int32)
    for ini in range(0, n, bloco):
        fim = min(ini + bloco, n)
        p = pixels[ini:fim][:, None, :]        # (b, 1, 3)
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
# 6. METRICAS: ERRO DE QUANTIZACAO E ERRO TOPOLOGICO
# ==============================================================================
def erro_quantizacao(pixels, paleta, indices=None):
    """Erro de quantizacao = distancia MEDIA (euclidiana) entre cada pixel e a
    cor do seu neuronio vencedor.

    E a medida de FIDELIDADE: quanto menor, mais parecida a imagem quantizada
    fica da original. Reportamos a distancia real (com raiz) para ter unidade
    interpretavel na escala [0, 1] de cada canal.
    """
    pixels = np.asarray(pixels, dtype=np.float32)
    paleta = np.asarray(paleta, dtype=np.float32)
    if indices is None:
        indices = _bmu_indices(pixels, paleta)
    diffs = pixels - paleta[indices]
    dist = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.mean(dist))


def erro_topologico_som(pixels, som):
    """Erro topologico do SOM = fracao de amostras cujo 1o e 2o BMU NAO sao
    vizinhos no grid.

    INTUICAO: se o mapa preserva bem a topologia, os dois neuronios mais
    proximos de uma amostra devem estar lado a lado no grid. Quando o 1o e o 2o
    colocado ficam distantes no grid, houve uma "dobra" no mapa (topologia mal
    preservada). Quanto menor o erro, melhor a preservacao topologica.
    """
    pesos = som.get_weights()                 # (lado, lado, 3)
    lado_x, lado_y, _ = pesos.shape
    grade = pesos.reshape(-1, 3)              # (K, 3)
    pixels = np.asarray(pixels, dtype=np.float32)

    violacoes = 0
    bloco = 50_000
    for ini in range(0, len(pixels), bloco):
        p = pixels[ini:ini + bloco][:, None, :]           # (b,1,3)
        d2 = np.sum((p - grade[None, :, :]) ** 2, axis=2)  # (b,K)
        # indices dos 2 mais proximos em cada linha (ordem interna nao garantida
        # pelo argpartition, por isso reordenamos pelo valor logo abaixo).
        dois = np.argpartition(d2, 1, axis=1)[:, :2]
        # coordenadas 2D no grid a partir do indice achatado.
        for linha_idx, par in enumerate(dois):
            d_par = d2[linha_idx, par]
            if d_par[0] <= d_par[1]:
                b1, b2 = par[0], par[1]
            else:
                b1, b2 = par[1], par[0]
            x1, y1 = divmod(b1, lado_y)
            x2, y2 = divmod(b2, lado_y)
            # vizinhos no grid = distancia de Chebyshev <= 1 (inclui diagonais).
            if max(abs(x1 - x2), abs(y1 - y2)) > 1:
                violacoes += 1
    return violacoes / len(pixels)


def erro_topologico_gng(pixels, gng):
    """Erro topologico do GNG = fracao de amostras cujo 1o e 2o BMU NAO estao
    ligados por uma ARESTA do grafo.

    No GNG a topologia sao as arestas aprendidas (nao um grid fixo). Se os dois
    prototipos mais proximos de uma amostra nao estao conectados, a topologia
    nao capturou bem aquela vizinhanca. E a definicao padrao de erro topologico
    para GNG e o analogo direto do criterio usado no SOM.
    """
    paleta = gng.paleta()
    pixels = np.asarray(pixels, dtype=np.float32)
    arestas = gng.arestas

    def conectados(i, j):
        chave = (i, j) if i < j else (j, i)
        return chave in arestas

    violacoes = 0
    bloco = 50_000
    for ini in range(0, len(pixels), bloco):
        p = pixels[ini:ini + bloco][:, None, :]
        d2 = np.sum((p - paleta[None, :, :]) ** 2, axis=2)
        dois = np.argpartition(d2, 1, axis=1)[:, :2]
        for linha_idx, par in enumerate(dois):
            d_par = d2[linha_idx, par]
            if d_par[0] <= d_par[1]:
                b1, b2 = int(par[0]), int(par[1])
            else:
                b1, b2 = int(par[1]), int(par[0])
            if not conectados(b1, b2):
                violacoes += 1
    return violacoes / len(pixels)


# ==============================================================================
# 7. VISUALIZACAO: ORIGINAL vs QUANTIZADA + PALETA EXTRAIDA
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


# ==============================================================================
# 8. EXECUCAO PRINCIPAL
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Quantizacao de cores com SOM e GNG (Projeto 2)."
    )
    parser.add_argument("--imagem", required=True,
                        help="Caminho da imagem de entrada (PNG/JPG).")
    parser.add_argument("--amostras", type=int, default=20000,
                        help="Nº de pixels amostrados para TREINO (default 20000).")
    parser.add_argument("--saida", default="resultados",
                        help="Pasta onde salvar as figuras de resultado.")
    args = parser.parse_args()

    os.makedirs(args.saida, exist_ok=True)

    # ----- 1. EXTRACAO --------------------------------------------------------
    print(f"[INFO] Carregando imagem: {args.imagem}")
    pixels, forma = carregar_pixels(args.imagem)
    original_uint8 = (pixels.reshape(forma) * 255).astype(np.uint8)
    print(f"       Dimensoes: {forma[1]}x{forma[0]} px  "
          f"({len(pixels):,} pixels, {len(np.unique(pixels, axis=0)):,} cores unicas)")

    dados_treino = amostrar_pixels(pixels, args.amostras)
    print(f"       Amostra de treino: {len(dados_treino):,} pixels\n")

    resultados = []  # linhas da tabela final

    # ----- 2/3/4/5. SOM: grids 4x4, 8x8, 16x16 -------------------------------
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
        et = erro_topologico_som(dados_treino, som)  # topologico na amostra (rapido)
        quant = reconstruir_imagem(idx, pal, forma)
        nome = os.path.join(args.saida, f"som_{lado}x{lado}.png")
        salvar_comparacao(nome, original_uint8, quant, pal,
                          titulo=f"SOM {lado}x{lado}", n_cores=k)
        dt = time.time() - t0
        print(f"  Grid {lado}x{lado:<2} (K={k:<3})  "
              f"EQ={eq:.4f}  ET={et:.4f}  tempo={dt:.1f}s  -> {nome}")
        resultados.append(("SOM", f"{lado}x{lado}", k, eq, et, dt))

    # ----- 2/3/4/5. GNG: 16, 64, 256 neuronios -------------------------------
    print("\n" + "=" * 74)
    print("GNG (Growing Neural Gas) - implementacao propria")
    print("=" * 74)
    for k in (16, 64, 256):
        t0 = time.time()
        gng = GrowingNeuralGas(n_neuronios_max=k, max_epocas=20)
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
    print(f"\n[OK] Figuras salvas em: {args.saida}/")


if __name__ == "__main__":
    main()
