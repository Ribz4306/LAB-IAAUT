import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------------- Valores (alterar aqui) ----------------
Rout1 = 2.7e3     # resultado do ex. 1.2 (confirmar com quem fez o ex. 1)
R5 = 1.2e3
R6 = 330e3
C3 = 10e-9
C4 = 10e-9


def rauch_bp(R1, R2=R5, Rf=R6, Ca=C3, Cb=C4):
    """Devolve num, den de T(s) para o passa-banda de Rauch."""
    num = [-1.0 / (R1 * Cb), 0.0]
    den = [1.0, (Ca + Cb) / (Rf * Ca * Cb), (1 / R1 + 1 / R2) / (Rf * Ca * Cb)]
    return num, den


def parametros(R1, R2=R5, Rf=R6, Ca=C3, Cb=C4):
    """f0, Q, largura de banda e ganho no centro (fórmulas analíticas)."""
    w0 = np.sqrt((1 / R1 + 1 / R2) / (Rf * Ca * Cb))
    bw = (Ca + Cb) / (Rf * Ca * Cb)           # rad/s  (= w0/Q)
    K = (Rf / R1) * Ca / (Ca + Cb)             # |T(jw0)|
    Q = w0 / bw
    f0 = w0 / (2 * np.pi)
    # frequências de corte a -3 dB de um passa-banda de 2.ª ordem
    fL = f0 * (np.sqrt(1 + 1 / (4 * Q**2)) - 1 / (2 * Q))
    fH = f0 * (np.sqrt(1 + 1 / (4 * Q**2)) + 1 / (2 * Q))
    return dict(f0=f0, Q=Q, B=bw / (2 * np.pi), fL=fL, fH=fH,
                G0=K, G0_dB=20 * np.log10(K))


# ---------------- 1) Bode com Rout1 ----------------
num, den = rauch_bp(Rout1)
sys = signal.TransferFunction(num, den)
f = np.logspace(1, 5, 2000)                   # 10 Hz -> 100 kHz
w, mag, fase = signal.bode(sys, 2 * np.pi * f)

p = parametros(Rout1)
print("=== Filtro com Rout1 = %.0f Ohm ===" % Rout1)
print("f0 = %.1f Hz | Q = %.2f | LB3dB = %.1f Hz | G0 = %.1f (%.2f dB)"
      % (p["f0"], p["Q"], p["B"], p["G0"], p["G0_dB"]))
print("fL = %.1f Hz | fH = %.1f Hz" % (p["fL"], p["fH"]))


def ganho_fase(fk):
    """Ganho (dB) e fase (graus, entre -360 e 0) à frequência fk."""
    _, H = signal.freqresp(sys, [2 * np.pi * fk])
    fase = np.degrees(np.angle(H[0]))
    if fase > 0:                 # pôr a fase contínua: -90 -> -180 -> -270
        fase -= 360
    return 20 * np.log10(abs(H[0])), fase


# Tabela às frequências pedidas no 2.3 (para a coluna "Teórico")
print("\n  f (Hz)       |T| (dB)   fase (graus)")
for fk in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
    g, ph = ganho_fase(fk)
    print("%10d     %7.2f   %8.1f" % (fk, g, ph))
for nome in ["fL", "f0", "fH"]:
    g, ph = ganho_fase(p[nome])
    print("%2s=%6.1f     %7.2f   %8.1f" % (nome, p[nome], g, ph))

fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
a1.semilogx(f, mag)
a1.axhline(p["G0_dB"] - 3, ls="--", lw=0.8, color="gray")
a1.set_ylabel("Ganho (dB)")
a1.set_title("Bode - filtro de Rauch (Rout1 = %.1f k$\\Omega$)" % (Rout1 / 1e3))
a1.grid(True, which="both", alpha=0.3)
a2.semilogx(f, fase)
a2.set_ylabel("Fase (°)")
a2.set_xlabel("Frequência (Hz)")
a2.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("bode_rauch.png", dpi=150)

# ---------------- 2) Efeito de Rout1 ----------------
print("\n=== Efeito de Rout1 ===")
fig2, (b1, b2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
for R1 in [1e3, 2.7e3, 5e3, 10e3]:
    s_ = signal.TransferFunction(*rauch_bp(R1))
    _, m, ph = signal.bode(s_, 2 * np.pi * f)
    q = parametros(R1)
    lbl = "Rout1 = %.1f k$\\Omega$" % (R1 / 1e3)
    b1.semilogx(f, m, label=lbl)
    b2.semilogx(f, ph, label=lbl)
    print("Rout1 = %6.0f -> f0 = %6.1f Hz, G0 = %5.2f dB, LB = %.1f Hz"
          % (R1, q["f0"], q["G0_dB"], q["B"]))
b1.set_ylabel("Ganho (dB)")
b1.set_title("Influência de Rout1 na resposta do filtro")
b1.legend()
b1.grid(True, which="both", alpha=0.3)
b2.set_ylabel("Fase (°)")
b2.set_xlabel("Frequência (Hz)")
b2.grid(True, which="both", alpha=0.3)
fig2.tight_layout()
fig2.savefig("bode_rauch_Rout1.png", dpi=150)

plt.show()
