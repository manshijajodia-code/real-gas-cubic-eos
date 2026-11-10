from __future__ import annotations
import argparse
import math
import sys
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
R_J = 8.314
R_V = 83.14

def _cp(A, B=0.0, C=0.0, D=0.0):
    return (A, B * 1e-3, C * 1e-6, D * 1e5)
DATABASE = {
    "methane": (0.012, 190.6, 45.99, _cp(1.702, 9.081, -2.164)),
    "ethane": (0.100, 305.3, 48.72, _cp(1.131, 19.225, -5.561)),
    "propane": (0.152, 369.8, 42.48, _cp(1.213, 28.785, -8.824)),
    "n-butane": (0.200, 425.1, 37.96, _cp(1.935, 36.915, -11.402)),
    "isobutane": (0.181, 408.1, 36.48, _cp(1.677, 37.853, -11.945)),
    "n-pentane": (0.252, 469.7, 33.70, _cp(2.464, 45.351, -14.111)),
    "n-hexane": (0.301, 507.6, 30.25, _cp(3.025, 53.722, -16.791)),
    "n-heptane": (0.350, 540.2, 27.40, _cp(3.570, 62.127, -19.486)),
    "n-octane": (0.400, 568.7, 24.90, _cp(4.108, 70.567, -22.208)),
    "ethylene": (0.087, 282.3, 50.40, _cp(1.424, 14.394, -4.392)),
    "propylene": (0.140, 365.6, 46.65, _cp(1.637, 22.706, -6.915)),
    "1-butene": (0.191, 420.0, 40.43, _cp(1.967, 31.630, -9.873)),
    "acetylene": (0.187, 308.3, 61.39, _cp(6.132, 1.952, 0.0, -1.299)),
    "benzene": (0.210, 562.2, 48.98, _cp(-0.206, 39.064, -13.301)),
    "toluene": (0.262, 591.8, 41.06, _cp(0.290, 47.052, -15.716)),
    "cyclohexane": (0.210, 553.6, 40.73, _cp(-3.876, 63.249, -20.928)),
    "methanol": (0.564, 512.6, 80.97, _cp(2.211, 12.216, -3.450)),
    "ethanol": (0.645, 513.9, 61.48, _cp(3.518, 20.001, -6.002)),
    "argon": (0.000, 150.9, 48.98, _cp(2.5)),
    "hydrogen": (-0.216, 33.19, 13.13, _cp(3.249, 0.422, 0.0, 0.083)),
    "oxygen": (0.022, 154.6, 50.43, _cp(3.639, 0.506, 0.0, -0.227)),
    "nitrogen": (0.038, 126.2, 34.00, _cp(3.280, 0.593, 0.0, 0.040)),
    "air": (0.035, 132.2, 37.45, _cp(3.355, 0.575, 0.0, -0.016)),
    "carbon monoxide": (0.048, 132.9, 34.99, _cp(3.376, 0.557, 0.0, -0.031)),
    "carbon dioxide": (0.224, 304.2, 73.83, _cp(5.457, 1.045, 0.0, -1.157)),
    "hydrogen sulfide": (0.094, 373.5, 89.63, _cp(3.931, 1.490, 0.0, -0.232)),
    "sulfur dioxide": (0.245, 430.8, 78.84, _cp(5.699, 0.801, 0.0, -1.015)),
    "ammonia": (0.253, 405.7, 112.80, _cp(3.578, 3.020, 0.0, -0.186)),
    "water": (0.345, 647.1, 220.55, _cp(3.470, 1.450, 0.0, 0.121)),
}
ALIASES = {"co2": "carbon dioxide", "n2": "nitrogen", "o2": "oxygen", "h2": "hydrogen",
           "ch4": "methane", "c2h6": "ethane", "c3h8": "propane", "butane": "n-butane",
           "nh3": "ammonia", "h2o": "water", "so2": "sulfur dioxide", "h2s": "hydrogen sulfide",
           "co": "carbon monoxide", "ar": "argon", "pentane": "n-pentane", "hexane": "n-hexane",
           "heptane": "n-heptane", "octane": "n-octane", "iso-butane": "isobutane"}

@dataclass(frozen=True)
class Species:
    name: str
    omega: float
    Tc: float
    Pc: float
    cp: tuple = (3.5, 0.0, 0.0, 0.0)

    @staticmethod
    def get(name: str) -> "Species":
        key = name.strip().lower()
        key = ALIASES.get(key, key)
        if key not in DATABASE:
            raise KeyError(f"unknown species '{name}'. Known: {', '.join(sorted(DATABASE))}")
        w, Tc, Pc, cp = DATABASE[key]
        return Species(key, w, Tc, Pc, cp)

@dataclass(frozen=True)
class EOS:
    name: str
    eps: float
    sigma: float
    Omega: float
    Psi: float

    def alpha(self, Tr: float, w: float) -> float:
        if self.name == "vdW":
            return 1.0
        if self.name == "RK":
            return Tr ** -0.5
        k = self.kappa(w)
        return (1.0 + k * (1.0 - math.sqrt(Tr))) ** 2

    def kappa(self, w: float) -> float:
        if self.name == "SRK":
            return 0.480 + 1.574 * w - 0.176 * w * w
        if self.name == "PR":
            return 0.37464 + 1.54226 * w - 0.26992 * w * w
        raise ValueError

    def dlnalpha_dlnTr(self, Tr: float, w: float) -> float:
        if self.name == "vdW":
            return 0.0
        if self.name == "RK":
            return -0.5
        k = self.kappa(w)
        return -k * math.sqrt(Tr) / (1.0 + k * (1.0 - math.sqrt(Tr)))
_S2 = math.sqrt(2.0)
MODELS = {
    "vdW": EOS("vdW", 0.0, 0.0, 1.0 / 8.0, 27.0 / 64.0),
    "RK": EOS("RK", 0.0, 1.0, 0.08664, 0.42748),
    "SRK": EOS("SRK", 0.0, 1.0, 0.08664, 0.42748),
    "PR": EOS("PR", 1.0 - _S2, 1.0 + _S2, 0.07780, 0.45724),
}

def get_eos(name: str) -> EOS:
    for k, v in MODELS.items():
        if k.lower() == name.strip().lower():
            return v
    raise KeyError(f"unknown EOS '{name}'. Choose from {', '.join(MODELS)}")

class CubicEOS:
    def __init__(self, species: Species | str, eos: EOS | str = "PR"):
        self.sp = Species.get(species) if isinstance(species, str) else species
        self.eos = get_eos(eos) if isinstance(eos, str) else eos
        self._psat_cache: dict = {}

    @property
    def b(self) -> float:
        return self.eos.Omega * R_V * self.sp.Tc / self.sp.Pc

    def a(self, T: float) -> float:
        Tr = T / self.sp.Tc
        return self.eos.Psi * self.eos.alpha(Tr, self.sp.omega) * R_V ** 2 * self.sp.Tc ** 2 / self.sp.Pc

    def beta_q(self, T: float, P: float) -> tuple[float, float]:
        Tr, Pr = T / self.sp.Tc, P / self.sp.Pc
        beta = self.eos.Omega * Pr / Tr
        q = self.eos.Psi * self.eos.alpha(Tr, self.sp.omega) / (self.eos.Omega * Tr)
        return beta, q

    def pressure(self, T: float, V: float) -> float:
        e, s, b = self.eos.eps, self.eos.sigma, self.b
        return R_V * T / (V - b) - self.a(T) / ((V + e * b) * (V + s * b))

    def z_roots(self, T: float, P: float) -> np.ndarray:
        beta, q = self.beta_q(T, P)
        e, s = self.eos.eps, self.eos.sigma
        poly = np.polymul(np.polymul([1.0, -(1.0 + beta)], [1.0, e * beta]), [1.0, s * beta])
        poly = np.polyadd(poly, [q * beta, -q * beta * beta])
        r = np.roots(poly)
        real = np.sort([x.real for x in r if abs(x.imag) < 1e-9 * max(1.0, abs(x.real))])
        real = [x for x in real if x > beta * (1 + 1e-12)]
        out = []
        for z in real:
            for _ in range(5):
                f = np.polyval(poly, z)
                df = np.polyval(np.polyder(poly), z)
                if df == 0:
                    break
                z -= f / df
            out.append(z)
        return np.array(sorted(set(round(z, 14) for z in out)))

    def volumes(self, T: float, P: float) -> np.ndarray:
        return self.z_roots(T, P) * R_V * T / P

    def _I(self, Z: float, beta: float) -> float:
        e, s = self.eos.eps, self.eos.sigma
        if abs(s - e) < 1e-12:
            return beta / (Z + e * beta)
        return math.log((Z + s * beta) / (Z + e * beta)) / (s - e)

    def residual(self, T: float, P: float, Z: float) -> dict:
        beta, q = self.beta_q(T, P)
        I = self._I(Z, beta)
        dl = self.eos.dlnalpha_dlnTr(T / self.sp.Tc, self.sp.omega)
        lnphi = Z - 1.0 - math.log(Z - beta) - q * I
        HR_RT = Z - 1.0 + (dl - 1.0) * q * I
        SR_R = math.log(Z - beta) + dl * q * I
        return {"Z": Z, "V": Z * R_V * T / P, "lnphi": lnphi, "phi": math.exp(lnphi),
                "f": P * math.exp(lnphi), "HR": HR_RT * R_J * T, "SR": SR_R * R_J,
                "GR": lnphi * R_J * T}

    def dH_ig(self, T0: float, T: float) -> float:
        A, B, C, D = self.sp.cp
        return R_J * (A * (T - T0) + B / 2 * (T * T - T0 * T0) + C / 3 * (T ** 3 - T0 ** 3)
                      + D * (1.0 / T0 - 1.0 / T))

    def dS_ig(self, T0: float, T: float, P0: float, P: float) -> float:
        A, B, C, D = self.sp.cp
        return R_J * (A * math.log(T / T0) + B * (T - T0) + C / 2 * (T * T - T0 * T0)
                      + D / 2 * (1.0 / (T0 * T0) - 1.0 / (T * T))) - R_J * math.log(P / P0)

    def psat(self, T: float) -> float | None:
        if T >= self.sp.Tc:
            return None
        if T in self._psat_cache:
            return self._psat_cache[T]
        Pc = self.sp.Pc

        def g(P):
            z = self.z_roots(T, P)
            if len(z) < 2:
                return None
            return self.residual(T, P, z[0])["lnphi"] - self.residual(T, P, z[-1])["lnphi"]
        grid = np.geomspace(1e-6 * Pc, 1.2 * Pc, 1500)
        vals = [(P, g(P)) for P in grid]
        pts = [(P, v) for P, v in vals if v is not None]
        for (P1, v1), (P2, v2) in zip(pts[:-1], pts[1:]):
            if v1 * v2 < 0:
                p = float(brentq(lambda p: g(p) if g(p) is not None else 0.0, P1, P2, xtol=1e-12, rtol=1e-12))
                self._psat_cache[T] = p
                return p
        self._psat_cache[T] = None
        return None

    def state(self, T: float, P: float, phase: str = "auto", T0: float = 298.15, P0: float = 1.0) -> dict:
        z = self.z_roots(T, P)
        if len(z) == 0:
            raise RuntimeError("no physical root found")
        Ps = self.psat(T)
        if phase == "vapor":
            Z = z[-1]
        elif phase == "liquid":
            Z = z[0]
        elif len(z) == 1:
            Z = z[0]
        else:
            lo = self.residual(T, P, z[0])["lnphi"]
            hi = self.residual(T, P, z[-1])["lnphi"]
            Z = z[0] if lo < hi else z[-1]
        res = self.residual(T, P, Z)
        if T > self.sp.Tc:
            label = "supercritical fluid" if P > self.sp.Pc else "gas (T > Tc)"
        elif Ps is None:
            label = "-"
        else:
            label = "vapor" if P < Ps else "liquid"
            if phase in ("vapor", "liquid") and label != phase:
                label = f"{phase} (metastable, forced)"
        res.update({"T": T, "P": P, "phase": label, "roots": list(z), "Psat": Ps,
                    "H": self.dH_ig(T0, T) + res["HR"],
                    "S": self.dS_ig(T0, T, P0, P) + res["SR"],
                    "Hig": self.dH_ig(T0, T), "Sig": self.dS_ig(T0, T, P0, P)})
        return res

    def delta(self, T1, P1, T2, P2, phase1="auto", phase2="auto") -> dict:
        s1 = self.state(T1, P1, phase1)
        s2 = self.state(T2, P2, phase2)
        return {"dH": s2["H"] - s1["H"], "dS": s2["S"] - s1["S"], "s1": s1, "s2": s2}

    def latent_heat(self, T: float) -> dict | None:
        Ps = self.psat(T)
        if Ps is None:
            return None
        z = self.z_roots(T, Ps)
        l, v = self.residual(T, Ps, z[0]), self.residual(T, Ps, z[-1])
        return {"Psat": Ps, "Hvap": float(v["HR"] - l["HR"]), "Svap": float(v["SR"] - l["SR"]),
                "Vl": float(l["V"]), "Vv": float(v["V"])}

def plot_isotherms(m: CubicEOS, Trs=(0.8, 0.9, 1.0, 1.1, 1.3), path="isotherms.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    b = m.b
    for Tr in Trs:
        T = Tr * m.sp.Tc
        V = np.geomspace(1.02 * b, 40 * m.sp.Tc * R_V / m.sp.Pc, 600)
        ax.plot(V, [m.pressure(T, v) for v in V], label=f"Tr = {Tr}")
        Ps = m.psat(T)
        if Ps:
            z = m.z_roots(T, Ps)
            ax.plot(z[[0, -1]] * R_V * T / Ps, [Ps, Ps], "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-0.3 * m.sp.Pc, 2.0 * m.sp.Pc)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("V  (cm$^3$/mol)")
    ax.set_ylabel("P  (bar)")
    ax.set_title(f"{m.sp.name}, {m.eos.name}: isotherms (dashed: equal-area line at P$^{{sat}}$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def plot_Z(m: CubicEOS, Trs=(0.9, 1.0, 1.2, 1.5, 2.0), Prmax=8.0, path="zchart.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for Tr in Trs:
        T = Tr * m.sp.Tc
        Ps = m.psat(T)
        xs, zs = [], []
        for Pr in np.linspace(0.02, Prmax, 300):
            P = Pr * m.sp.Pc
            if Ps and abs(P - Ps) / Ps < 0.0:
                continue
            xs.append(Pr)
            zs.append(m.state(T, P)["Z"])
        ax.plot(xs, zs, label=f"Tr = {Tr}")
    ax.set_xlabel("P$_r$")
    ax.set_ylabel("Z")
    ax.set_title(f"{m.sp.name}, {m.eos.name}: compressibility factor")
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def plot_props(m: CubicEOS, Tr=0.9, path="props.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = Tr * m.sp.Tc
    Ps = m.psat(T)
    Ps = Ps or 0.5 * m.sp.Pc
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    P = np.linspace(0.02 * m.sp.Pc, 1.6 * m.sp.Pc, 400)
    rows = [m.state(T, p) for p in P]
    axs[0].plot(P, [r["lnphi"] for r in rows]); axs[0].set_ylabel("ln φ")
    axs[1].plot(P, [r["HR"] / 1000 for r in rows]); axs[1].set_ylabel("H$^R$ (kJ/mol)")
    axs[2].plot(P, [r["SR"] for r in rows]); axs[2].set_ylabel("S$^R$ (J/mol K)")
    for a in axs:
        a.axvline(Ps, color="gray", ls="--", lw=0.8)
        a.set_xlabel("P (bar)")
    fig.suptitle(f"{m.sp.name}, {m.eos.name}, T = {T:.1f} K (Tr = {Tr}); dashed = P$^{{sat}}$", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def report(m: CubicEOS, T: float, P: float, phase="auto", T0=298.15, P0=1.0) -> str:
    s = m.state(T, P, phase, T0, P0)
    a, b = m.a(T), m.b
    lines = [
        f"{m.sp.name}  |  {m.eos.name} EOS  |  T = {T:g} K,  P = {P:g} bar",
        f"  Tc = {m.sp.Tc} K, Pc = {m.sp.Pc} bar, omega = {m.sp.omega}   (Tr = {T/m.sp.Tc:.4f}, Pr = {P/m.sp.Pc:.4f})",
        f"  a(T) = {a:.5g} bar cm6/mol2,  b = {b:.5g} cm3/mol",
        f"  Z roots: {', '.join(f'{z:.5f}' for z in s['roots'])}   -> volumes (cm3/mol): "
        f"{', '.join(f'{z*R_V*T/P:.4g}' for z in s['roots'])}",
    ]
    if s["Psat"]:
        lines.append(f"  Psat({T:g} K) = {s['Psat']:.5g} bar   -> state is {s['phase']}")
    else:
        lines.append(f"  state: {s['phase']}")
    lines += [
        f"  Z = {s['Z']:.5f},  V = {s['V']:.5g} cm3/mol,  density = {1e6/ s['V']/1e3:.4g} mol/L",
        f"  ln(phi) = {s['lnphi']:.5f},  phi = {s['phi']:.5f},  f = {s['f']:.5g} bar",
        f"  H^R = {s['HR']:.1f} J/mol,  S^R = {s['SR']:.3f} J/(mol K),  G^R = {s['GR']:.1f} J/mol",
        f"  H = {s['H']:.1f} J/mol,  S = {s['S']:.3f} J/(mol K)   "
        f"(reference: ideal gas at {T0:g} K, {P0:g} bar, H = S = 0)",
    ]
    return "\n".join(lines)

def _ask(prompt, default=None, cast=str):
    txt = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if not txt:
        if default is None:
            raise ValueError("a value is required")
        return default
    return cast(txt)

def menu():
    print("Cubic EOS calculator: real-gas PVT, H, S and fugacity")
    sp, eos = "n-butane", "PR"
    while True:
        print(f"\nCurrent: {sp}, {eos}")
        print(" 1 species / EOS   2 state at T,P (Z, V, H, S, fugacity)   3 P at T,V\n"
              " 4 vapour pressure Psat(T) and latent heat   5 property change between two states\n"
              " 6 compare the four EOS at T,P   7 plots   8 species list   0 quit")
        try:
            c = _ask("choice", "2")
            if c == "0":
                return
            if c == "1":
                sp = _ask("species", sp)
                eos = _ask("EOS (vdW/RK/SRK/PR)", eos)
                CubicEOS(sp, eos)
            elif c == "2":
                m = CubicEOS(sp, eos)
                print(report(m, _ask("T (K)", cast=float), _ask("P (bar)", cast=float)))
            elif c == "3":
                m = CubicEOS(sp, eos)
                T, V = _ask("T (K)", cast=float), _ask("V (cm3/mol)", cast=float)
                print(f"P = {m.pressure(T, V):.6g} bar")
            elif c == "4":
                m = CubicEOS(sp, eos)
                T = _ask("T (K)", cast=float)
                L = m.latent_heat(T)
                print("T >= Tc: no vapour pressure" if L is None else
                      f"Psat = {L['Psat']:.5g} bar, Hvap = {L['Hvap']:.1f} J/mol, Svap = {L['Svap']:.3f} J/mol K, "
                      f"Vl = {L['Vl']:.4g}, Vv = {L['Vv']:.4g} cm3/mol")
            elif c == "5":
                m = CubicEOS(sp, eos)
                T1, P1 = _ask("T1 (K)", cast=float), _ask("P1 (bar)", cast=float)
                T2, P2 = _ask("T2 (K)", cast=float), _ask("P2 (bar)", cast=float)
                d = m.delta(T1, P1, T2, P2)
                print(f"state 1: {d['s1']['phase']}, Z = {d['s1']['Z']:.4f}; state 2: {d['s2']['phase']}, Z = {d['s2']['Z']:.4f}")
                print(f"dH = {d['dH']:.1f} J/mol,  dS = {d['dS']:.3f} J/(mol K)")
            elif c == "6":
                T, P = _ask("T (K)", cast=float), _ask("P (bar)", cast=float)
                print(f"{'EOS':5s}{'Z':>9s}{'V cm3/mol':>12s}{'phi':>9s}{'f bar':>10s}{'HR J/mol':>11s}{'SR J/molK':>11s}")
                for k in MODELS:
                    s = CubicEOS(sp, k).state(T, P)
                    print(f"{k:5s}{s['Z']:9.4f}{s['V']:12.4g}{s['phi']:9.4f}{s['f']:10.4g}{s['HR']:11.1f}{s['SR']:11.3f}")
            elif c == "7":
                m = CubicEOS(sp, eos)
                print("saved:", plot_isotherms(m), plot_Z(m), plot_props(m))
            elif c == "8":
                print(", ".join(sorted(DATABASE)))
        except (ValueError, KeyError, RuntimeError) as e:
            print("Error:", e)
        except (EOFError, KeyboardInterrupt):
            return

def main(argv=None):
    ap = argparse.ArgumentParser(description="Real-gas PVT, H, S, fugacity from cubic EOS")
    ap.add_argument("species", nargs="?", help="e.g. n-butane, CO2, methane")
    ap.add_argument("-T", type=float, help="temperature, K")
    ap.add_argument("-P", type=float, help="pressure, bar")
    ap.add_argument("--eos", default="PR", help="vdW, RK, SRK or PR (default PR)")
    ap.add_argument("--phase", default="auto", choices=["auto", "vapor", "liquid"])
    ap.add_argument("--T0", type=float, default=298.15, help="reference temperature for H,S (K)")
    ap.add_argument("--P0", type=float, default=1.0, help="reference pressure for H,S (bar)")
    ap.add_argument("--psat", action="store_true", help="also print Psat and latent heat at T")
    ap.add_argument("--compare", action="store_true", help="compare all four EOS")
    ap.add_argument("--plots", action="store_true", help="write isotherms.png, zchart.png, props.png")
    a = ap.parse_args(argv)
    if a.species is None:
        return menu()
    if a.T is None or a.P is None:
        ap.error("give both -T and -P")
    names = list(MODELS) if a.compare else [a.eos]
    for n in names:
        m = CubicEOS(a.species, n)
        print(report(m, a.T, a.P, a.phase, a.T0, a.P0))
        if a.psat:
            L = m.latent_heat(a.T)
            print("  (no vapour pressure above Tc)" if L is None else
                  f"  Psat = {L['Psat']:.5g} bar, Hvap = {L['Hvap']:.1f} J/mol, Svap = {L['Svap']:.3f} J/mol K")
        print()
    if a.plots:
        m = CubicEOS(a.species, a.eos)
        print("saved:", plot_isotherms(m), plot_Z(m), plot_props(m))
if __name__ == "__main__":
    main()
