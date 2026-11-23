# Projeto 2 — Quantização de Cores com Aprendizado Não Supervisionado (SOM e GNG)

**Disciplina:** Introdução às Redes Neurais Artificiais
**Autor:** Henrique Crepaldi (RA: 120410)
**Código:** `quantizacao_cores_project2.py` (NumPy + Pillow + MiniSom)

> **Objetivo.** Reduzir drasticamente a paleta de cores de uma imagem (compressão/quantização)
> usando duas redes de **aprendizado competitivo não supervisionado** — **SOM** (Self-Organizing
> Map) e **GNG** (Growing Neural Gas) — e comparar as duas quanto à **fidelidade** (erro de
> quantização) e à **preservação topológica** (erro topológico).

---

## 1. A tarefa e por que não há vazamento de dados aqui

Cada pixel da imagem é tratado como uma **amostra 3D** com as características (R, G, B). A rede
aprende um conjunto pequeno de **cores-protótipo** (16, 64 ou 256) que resumem a nuvem de cores
da imagem. Depois, cada pixel é substituído pela cor do seu **neurônio vencedor (BMU)**.

**Caminho adotado (Caminho A):** treina-se a rede nos pixels de **uma** imagem e quantiza-se
**essa mesma** imagem. Isso é **compressão**, não generalização para imagens novas. Logo, **não
existe conjunto de teste nem risco de vazamento** nesta tarefa — treinar e inferir na mesma
imagem é a própria natureza da quantização. (A variante "treinar na imagem A, aplicar na B",
onde o conceito de vazamento voltaria a fazer sentido, fica como trabalho futuro — Caminho B.)

> **Nota de reprodutibilidade / validação.** As imagens usadas para validar o pipeline
> (`Out-Sem-1_01.jpg` e `Out-Sem-1_02.jpg`, dois flyers do pub, 1350×1080 px) são imagens que
> **não** entram no projeto final — foram usadas apenas para confirmar que o código roda e
> produz resultados coerentes. Nenhuma imagem sintética foi usada.

---

## 2. Pipeline (conforme o enunciado)

| Etapa | O que o código faz | Função no script |
|-------|--------------------|------------------|
| **Extração** | Carrega a imagem, extrai todos os pixels, normaliza RGB para [0, 1]. | `carregar_pixels` |
| **Treinamento** | SOM (grids 4×4, 8×8, 16×16) e GNG (16, 64, 256 neurônios). | `treinar_som`, `GrowingNeuralGas.treinar` |
| **Inferência** | Para cada pixel, acha o BMU e devolve o índice do neurônio. | `_bmu_indices`, `GrowingNeuralGas.quantizar` |
| **Reconstrução** | Troca a cor de cada pixel pela cor dos pesos do BMU e remonta a matriz. | `reconstruir_imagem` |
| **Avaliação** | Erro de quantização (EQ) e erro topológico (ET). | `erro_quantizacao`, `erro_topologico_som`, `erro_topologico_gng` |

**Normalização [0, 1]:** os canais chegam como inteiros [0, 255]. Como as redes competitivas
usam **distâncias euclidianas** entre pesos e amostras, manter tudo em [0, 1] deixa a escala dos
três canais homogênea e dá significado estável às taxas de aprendizado.

**Amostragem de treino:** uma imagem 1350×1080 tem ~1,46 milhão de pixels. Treinar em todos é
desnecessário — uma amostra aleatória de 20 000 pixels já representa muito bem a distribuição de
cores. A **inferência** (trocar a cor) é feita em **todos** os pixels; só o **treino** usa a
amostra.

---

## 3. As duas redes, lado a lado

| Aspecto | **SOM** (MiniSom) | **GNG** (implementação própria) |
|---------|-------------------|--------------------------------|
| Topologia | **Fixa**: grid 2D definido a priori (ex.: 8×8). | **Cresce e se adapta**: grafo de arestas aprendido dos dados. |
| Nº de neurônios | Fixo (lado × lado). | Começa com 2 e **cresce** até o alvo (16/64/256). |
| Vizinhança | Gaussiana no grid, encolhe com o tempo. | Arestas criadas entre o 1º e o 2º BMU; envelhecem e morrem. |
| Atualização | Vencedor **e** vizinhos no grid. | Vencedor (passo `eps_b`) e vizinhos de grafo (passo `eps_n`). |

**Por que o GNG foi implementado do zero?** Não há biblioteca de GNG madura e mantida para o
Python moderno (as existentes dependem de versões antigas de TensorFlow/NumPy ou são protótipos
de um autor só). Além disso, o GNG é um algoritmo de poucos passos bem definidos (Fritzke,
1995); lê-lo direto no código é a melhor forma de **entendê-lo para a apresentação**, em vez de
depender de uma caixa-preta. O SOM, por sua vez, usa a **MiniSom** — biblioteca estável, pequena
e de código legível.

### 3.1. O algoritmo do GNG, passo a passo (como está no código)

Para cada pixel `x` apresentado (`GrowingNeuralGas._passo`):

1. Achar o **vencedor** `s1` e o **segundo** `s2` mais próximos de `x`.
2. **Envelhecer** todas as arestas que saem de `s1` (o tempo passa para as conexões dele).
3. Acumular o **erro** de `s1` com a distância² até `x` (regiões mal representadas acumulam erro).
4. **Mover** `s1` em direção a `x` (passo `eps_b`) e os vizinhos de `s1` um pouco menos (`eps_n`).
5. **Conectar** `s1` e `s2` (ou zerar a idade da aresta) — observar os dois juntos é evidência de
   vizinhança na topologia dos dados.
6. **Remover** arestas velhas (idade > `idade_max`) e neurônios que ficaram isolados.
7. A cada `lambda_insercao` amostras, **inserir** um novo neurônio na região de **maior erro**
   (`_inserir_neuronio`): nasce no ponto médio entre o neurônio de maior erro `q` e seu vizinho de
   maior erro `f`. É o mecanismo de **crescimento**.
8. **Decaimento** global do erro (todos "esquecem" um pouco do passado — o erro recente pesa mais).

---

## 4. Métricas

**Erro de quantização (EQ)** — mede **fidelidade**. É a distância euclidiana **média** entre
cada pixel e a cor do seu BMU, na escala [0, 1]. Quanto **menor**, mais parecida a imagem
quantizada fica da original.

**Erro topológico (ET)** — mede **preservação da topologia**. É a **fração de amostras cujo 1º e
2º BMU não são vizinhos**:
- **SOM:** vizinhos = adjacentes no grid (distância de Chebyshev ≤ 1, inclui diagonais).
- **GNG:** vizinhos = ligados por uma **aresta** do grafo aprendido.

Se o 1º e o 2º colocados não são vizinhos, houve uma "dobra" no mapa (SOM) ou uma vizinhança que
o grafo não capturou (GNG). Quanto **menor**, melhor a preservação topológica. É o mesmo
critério conceitual nas duas redes, o que torna a comparação justa.

---

## 5. Resultados (validação nos dois flyers)

Treino com 20 000 pixels amostrados; inferência em todos os 1 458 000 pixels. `SEED = 42`.

### 5.1. Imagem 1 — Radio Galena (`Out-Sem-1_01.jpg`, 131 548 cores únicas)

| Rede | Config | K | EQ (↓) | ET (↓) | Tempo (s) |
|------|--------|---|--------|--------|-----------|
| SOM  | 4×4    | 16  | 0.0641 | 0.0114 | 1.2  |
| SOM  | 8×8    | 64  | 0.0496 | 0.0428 | 3.0  |
| SOM  | 16×16  | 256 | 0.0297 | 0.0547 | 10.0 |
| GNG  | 16     | 16  | 0.0540 | **0.0000** | 9.0  |
| GNG  | 64     | 64  | **0.0249** | 0.0039 | 20.0 |
| GNG  | 256    | 256 | **0.0143** | 0.0056 | 60.8 |

### 5.2. Imagem 2 — Velotroll (`Out-Sem-1_02.jpg`, 198 031 cores únicas)

| Rede | Config | K | EQ (↓) | ET (↓) | Tempo (s) |
|------|--------|---|--------|--------|-----------|
| SOM  | 4×4    | 16  | 0.0946 | 0.0026 | 1.2  |
| SOM  | 8×8    | 64  | 0.0716 | 0.0049 | 3.0  |
| SOM  | 16×16  | 256 | 0.0515 | 0.0090 | 9.9  |
| GNG  | 16     | 16  | 0.0732 | 0.0055 | 8.9  |
| GNG  | 64     | 64  | **0.0351** | 0.0020 | 20.6 |
| GNG  | 256    | 256 | **0.0199** | 0.0041 | 63.4 |

---

## 6. Interpretação dos resultados

1. **Mais neurônios → menor EQ, em ambas as redes.** Mais cores-protótipo = reconstrução mais
   fiel. O padrão é monotônico nas duas imagens, como esperado.

2. **GNG vence o SOM em EQ para o mesmo K.** Ex.: na imagem 1, K=256 → GNG **0.0143** vs SOM
   **0.0297** (quase metade do erro). O SOM "gasta" neurônios para manter um **grid retangular
   fixo**, mesmo onde há poucas cores; o GNG não tem essa amarra e distribui os protótipos onde
   as cores realmente estão (regiões de maior erro recebem mais neurônios). Essa é a vantagem do
   crescimento adaptativo para **quantização pura**.

3. **GNG preserva melhor a topologia (ET menor).** As arestas do GNG se adaptam à nuvem de cores,
   enquanto o grid fixo do SOM precisa "dobrar" para caber nos dados, gerando mais violações —
   e **o ET do SOM cresce com o grid** (0.0114 → 0.0428 → 0.0547 na imagem 1), porque um grid
   maior tem mais chances de dobrar. No GNG o ET permanece baixo em todas as escalas.

4. **Custo computacional.** O GNG é mais caro (até ~60 s para K=256) porque é um laço explícito em
   Python com inserções incrementais; o SOM da MiniSom é vetorizado e mais rápido. Em troca, o
   GNG entrega melhor EQ e ET. É um **trade-off qualidade × tempo** que vale citar na banca.

5. **Imagem 2 teve EQ maior que a imagem 1** em quase todas as configs (ex.: SOM 4×4: 0.0946 vs
   0.0641). Faz sentido: a imagem 2 tem **mais cores únicas** (198k vs 131k) e mais variação, logo
   é mais difícil de resumir com poucas cores-protótipo.

---

## 7. Hiperparâmetros e como ajustá-los

**SOM** (`treinar_som`): `sigma` inicial = metade do lado do grid (começa cooperativo,
especializa com o tempo); `learning_rate` = 0.5; `num_iteracoes` = 5000. Aumentar as iterações
reduz um pouco o EQ, com retorno decrescente.

**GNG** (`GrowingNeuralGas`): os parâmetros mais sensíveis são:
- `eps_b` (passo do vencedor, 0.05) e `eps_n` (passo dos vizinhos, 0.006) — controlam velocidade
  × estabilidade do ajuste.
- `lambda_insercao` (200) — de quantas em quantas amostras um neurônio nasce; menor = cresce mais
  rápido.
- `idade_max` (50) — arestas mais velhas que isso morrem; menor = grafo mais "enxuto".
- `max_epocas` (20) — mais épocas refinam as posições após o crescimento parar.

---

## 8. Lições aprendidas

1. **Topologia fixa vs. adaptativa é o ponto central.** O SOM impõe um grid; o GNG descobre a
   estrutura. Para **quantização pura**, a liberdade do GNG rende menor EQ e menor ET. O SOM, em
   compensação, entrega um mapa 2D ordenado (útil quando se quer **visualizar** a organização das
   cores), e é bem mais rápido.
2. **EQ e ET medem coisas diferentes e complementares.** Uma rede pode ter EQ baixo e ainda assim
   "dobrar" a topologia. Reportar as duas evita conclusões enganosas.
3. **Implementar o GNG do zero valeu a pena** — além de contornar a falta de biblioteca madura,
   deixa cada passo (crescimento, envelhecimento de arestas, inserção por erro) explícito e
   defensável.
4. **A dificuldade da imagem importa.** Mais cores únicas ⇒ EQ maior para o mesmo K. A métrica
   deve sempre ser lida junto com a complexidade da entrada.

---

## 9. Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_projeto2.txt
python quantizacao_cores_project2.py --imagem caminho/da/sua/imagem.jpg --saida resultados
```

O script imprime a tabela de EQ/ET/tempo para as 6 configurações (SOM 4×4/8×8/16×16 e GNG
16/64/256) e salva, para cada uma, uma figura com **original × quantizada × paleta extraída** na
pasta de saída. Parâmetros úteis: `--amostras` (nº de pixels de treino, default 20000) e
`--saida` (pasta das figuras).
