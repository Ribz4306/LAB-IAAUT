"""
Parte I - Regressão Linear (modelo auto-regressivo)

    y(t) = y(t-1)^T β + β0

y(t) é a pose no instante t (66 valores: u e v dos 33 marcadores).
A pose seguinte é prevista a partir da pose anterior.
"""
import os
import sys
import numpy as np
import pandas as pd

PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_ACIMA = os.path.dirname(PASTA)
# Xsup.pkl e funcoes_aux podem estar nesta pasta ou na pasta de cima (LAB_2)
FICHEIRO_DADOS = next((p for p in (os.path.join(PASTA, "Xsup.pkl"),
                                   os.path.join(PASTA_ACIMA, "Xsup.pkl"))
                       if os.path.exists(p)), os.path.join(PASTA, "Xsup.pkl"))
sys.path.append(PASTA_ACIMA)



# R1.a) Estimação por equações normais

def estimar_coeficientes(X, Y):
    """
    Estima os coeficientes da regressão linear pelas equações normais:

        B = (X^T X)^(-1) X^T Y

    X : (N, d+1) com a coluna de 1s na PRIMEIRA posição.
    Y : (N, m) saídas (aqui m = 66, uma pose inteira).

    Devolve
        beta  : (d, m)  coeficientes de cada entrada
        beta0 : (m,)    intercept (um por saída)
    """
    XtX = X.T @ X
    XtY = X.T @ Y
    # resolver o sistema (X^T X)
    B = np.linalg.solve(XtX, XtY)
    beta0 = B[0, :]
    beta = B[1:, :]
    return beta, beta0


def prever(X, beta, beta0):
    """Aplica o modelo. X pode vir com ou sem a coluna de 1s."""
    if X.shape[1] == beta.shape[0] + 1:
        X = X[:, 1:]
    return X @ beta + beta0



# R1.b) Sequências do E5 do paciente 4, últimas 20% para teste

def carregar_dados(caminho=FICHEIRO_DADOS):
    return pd.read_pickle(caminho)


def separar_treino_teste(sequencias, frac_teste=0.2):
    """Separa mantendo a ordem original: as últimas frac_teste ficam para teste."""
    n_teste = int(np.ceil(frac_teste * len(sequencias)))
    n_treino = len(sequencias) - n_teste
    return sequencias[:n_treino], sequencias[n_treino:]


def selecionar_sequencias(df, paciente=4, exercicio="E5"):
    sel = df[(df["Patient_ID"] == paciente) & (df["Exercise_Id"] == exercicio)]
    # .tolist() mantém a ordem em que as sequências aparecem no ficheiro
    return sel["Skeleton_Sequence"].tolist()


# R1.c) e R1.d) Construção de X e Y

def construir_X_Y(sequencias):
    """
    Para cada sequência (T, 66):
        entradas  -> poses 0 .. T-2   (pose anterior)
        saídas    -> poses 1 .. T-1   (pose seguinte)
    Os pares são feitos DENTRO de cada sequência: a última pose de uma sequência
    nunca é usada para prever a primeira da seguinte.
    No fim acrescenta-se uma coluna de 1s (para o β0).
    """
    X = np.vstack([s[:-1] for s in sequencias])
    Y = np.vstack([s[1:] for s in sequencias])
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X, Y



# R1.e) SSE e R^2

def sse(Y, Y_hat):
    """Soma dos erros quadráticos sobre todas as poses e todas as 66 coordenadas."""
    return np.sum((Y - Y_hat) ** 2)


def r2(Y, Y_hat):
    """
    R^2 = 1 - SSE / SST
    SST mede a variação em torno da média de CADA coordenada (média por coluna),
    porque cada uma das 66 saídas tem a sua escala.
    """
    sst = np.sum((Y - Y.mean(axis=0)) ** 2)
    return 1 - sse(Y, Y_hat) / sst


def sse_por_keypoint(Y, Y_hat):
    """SSE de cada um dos 33 marcadores (soma do erro em u e em v)."""
    erro = (Y - Y_hat) ** 2                      # (N, 66)
    return erro.reshape(-1, 33, 2).sum(axis=(0, 2))  # (33,)


NOMES_KEYPOINTS = [
    "nariz", "olho_esq_interno", "olho_esq", "olho_esq_externo",
    "olho_dir_interno", "olho_dir", "olho_dir_externo",
    "orelha_esq", "orelha_dir", "boca_esq", "boca_dir",
    "ombro_esq", "ombro_dir", "cotovelo_esq", "cotovelo_dir",
    "punho_esq", "punho_dir", "mindinho_esq", "mindinho_dir",
    "indicador_esq", "indicador_dir", "polegar_esq", "polegar_dir",
    "anca_esq", "anca_dir", "joelho_esq", "joelho_dir",
    "tornozelo_esq", "tornozelo_dir", "calcanhar_esq", "calcanhar_dir",
    "pe_esq_dedo", "pe_dir_dedo",
]


def prever_sequencia(seq, beta, beta0):
    """
    Previsão um passo à frente para uma sequência inteira:
    a 1.ª pose é a verdadeira, cada pose seguinte é prevista a partir da pose
    verdadeira anterior.
    """
    prevista = np.empty_like(seq)
    prevista[0] = seq[0]
    prevista[1:] = prever(seq[:-1], beta, beta0)
    return prevista


def prever_sequencia_recursiva(seq, beta, beta0):
    """
    (Extra) Previsão em cadeia: só a 1.ª pose é verdadeira, cada previsão
    alimenta a seguinte. Mostra como o erro se acumula.
    """
    prevista = np.empty_like(seq)
    prevista[0] = seq[0]
    for t in range(1, len(seq)):
        prevista[t] = prever(prevista[t - 1:t], beta, beta0)[0]
    return prevista



# Parte 2 - Estimação e resultados

def main():
    df = carregar_dados()

    # R1.b
    sequencias = selecionar_sequencias(df, paciente=4, exercicio="E5")
    seq_treino, seq_teste = separar_treino_teste(sequencias, 0.2)
    print(f"Sequências E5 do paciente 4: {len(sequencias)} "
          f"-> treino {len(seq_treino)}, teste {len(seq_teste)}")

    # R1.c / R1.d
    X_treino, Y_treino = construir_X_Y(seq_treino)
    X_teste, Y_teste = construir_X_Y(seq_teste)
    print(f"X_treino {X_treino.shape}, Y_treino {Y_treino.shape}")
    print(f"X_teste  {X_teste.shape}, Y_teste  {Y_teste.shape}")

    # R2.a
    beta, beta0 = estimar_coeficientes(X_treino, Y_treino)
    print(f"\nR2.a) beta: {beta.shape}   beta0: {beta0.shape}")
    print(f"      parâmetros totais: {beta.size + beta0.size} "
          f"para {Y_treino.size} valores observados no treino")

    # R2.b
    Yh_treino = prever(X_treino, beta, beta0)
    print(f"\nR2.b) Treino: SSE = {sse(Y_treino, Yh_treino):.4f}   "
          f"R2 = {r2(Y_treino, Yh_treino):.4f}")

    # R2.c
    Yh_teste = prever(X_teste, beta, beta0)
    print(f"R2.c) Teste:  SSE = {sse(Y_teste, Yh_teste):.4f}   "
          f"R2 = {r2(Y_teste, Yh_teste):.4f}")
    # SSE médio por pose, para comparar conjuntos de tamanhos diferentes
    print(f"      SSE por pose: treino {sse(Y_treino, Yh_treino) / len(Y_treino):.5f}"
          f"  teste {sse(Y_teste, Yh_teste) / len(Y_teste):.5f}")

    sse_kp = sse_por_keypoint(Y_teste, Yh_teste)
    ordem = np.argsort(sse_kp)[::-1]
    print("\n      Keypoints com maior SSE no teste:")
    for j in ordem[:8]:
        print(f"        {j:2d} {NOMES_KEYPOINTS[j]:<16s} {sse_kp[j]:.4f}")

    # R2.d - vídeos lado a lado (verdadeira vs prevista)
    try:
        from funcoes_aux_proj_apr_supervisionada import visualize_sequences
    except ImportError as e:
        print(f"\nR2.d) visualize_sequences não disponível ({e}); vídeos não gerados.")
        return

    pasta_videos = os.path.join(PASTA, "videos_parte1")
    os.makedirs(pasta_videos, exist_ok=True)
    for i, seq in enumerate(seq_teste):
        videos = {
            "verdadeira": seq,
            "prevista": prever_sequencia(seq, beta, beta0),
            "prevista_recursiva": prever_sequencia_recursiva(seq, beta, beta0),
        }
        for nome, s in videos.items():
            html = visualize_sequences(s, title=f"Teste {i} - {nome}")
            with open(os.path.join(pasta_videos, f"teste{i}_{nome}.html"), "w") as f:
                f.write(html.data)
    print(f"\nR2.d) Vídeos guardados em {pasta_videos}")


if __name__ == "__main__":
    main()
