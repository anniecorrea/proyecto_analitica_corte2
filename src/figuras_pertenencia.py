"""figuras_pertenencia.py -- Matriz de confusion de la pertenencia anatomica (seccion 7). Autor: Fabian.

Una figura con dos paneles, en la misma escala de color, para separar las dos fuentes de error:

  Izquierda - SISTEMA COMPLETO (instancias predichas). Filas: clase real de cada fragmento. Columnas: clase que
  le dio la cabeza, mas una columna "no encontrado" para los fragmentos que la segmentacion no emparejo
  (IoU < 0.5). Esa columna es error de segmentacion, no de clasificacion.
  Derecha - MASCARAS REALES (solo clasificacion). La cabeza clasifica cada fragmento tal como esta anotado:
  mide lo que la cabeza sabe hacer sin depender del decoder.

Cada celda muestra el porcentaje de su FILA (que fraccion de los fragmentos de esa clase real termino en esa
columna; la diagonal es el recall de la clase) y el conteo. Se normaliza por fila porque las clases no tienen el
mismo numero de fragmentos: con conteos crudos, la clase mas frecuente dominaria el color.

Solo depende de numpy y matplotlib (se puede probar sin PyTorch).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ETIQUETAS = ("Sacro", "Coxal izq.", "Coxal der.")
# Rampa secuencial de un solo tono (azul, claro -> oscuro): mas oscuro = mayor fraccion de la fila.
RAMPA = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("azul_secuencial", RAMPA)
TINTA, TINTA_SECUNDARIA, SUPERFICIE = "#1f1f1d", "#5f5e5a", "#ffffff"


def _por_fila(conteos):
    """Normaliza cada fila a 1 (las filas sin fragmentos quedan en 0, no en NaN)."""
    conteos = np.asarray(conteos, dtype=float)
    total = conteos.sum(axis=1, keepdims=True)
    return np.divide(conteos, total, out=np.zeros_like(conteos), where=total > 0)


def _panel(ax, conteos, columnas, titulo, subtitulo, con_ylabel=True):
    frac = _por_fila(conteos)
    im = ax.imshow(frac, cmap=CMAP, vmin=0, vmax=1, aspect="equal")
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            color = SUPERFICIE if frac[i, j] > 0.55 else TINTA
            ax.text(j, i - 0.12, f"{frac[i, j]:.0%}", ha="center", va="center", fontsize=12,
                    fontweight="bold" if i == j else "normal", color=color)
            ax.text(j, i + 0.22, f"n={int(conteos[i, j]):,}".replace(",", "."), ha="center", va="center",
                    fontsize=8.5, color=color)
    ax.set_xticks(range(len(columnas)), columnas, fontsize=9.5, color=TINTA)
    ax.set_yticks(range(len(ETIQUETAS)), ETIQUETAS if con_ylabel else [""] * len(ETIQUETAS), fontsize=9.5, color=TINTA)
    ax.set_xlabel("Clase predicha por la cabeza", fontsize=9.5, color=TINTA_SECUNDARIA)
    if con_ylabel:   # el panel derecho tiene las mismas filas en el mismo orden: no las repite
        ax.set_ylabel("Clase real", fontsize=9.5, color=TINTA_SECUNDARIA)
    ax.set_title(titulo, fontsize=11, color=TINTA, loc="left", fontweight="bold", pad=34)
    ax.text(0, 1.03, subtitulo, transform=ax.transAxes, fontsize=8.5, color=TINTA_SECUNDARIA, va="bottom",
            linespacing=1.4)
    # Separacion de 2 px del color de la superficie entre celdas; sin marco ni marcas de eje.
    ax.set_xticks(np.arange(-0.5, frac.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, frac.shape[0]), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)
    return im


def figura_confusion_pertenencia(matriz_sistema, matriz_mascaras_reales, path, titulo="", f1_sistema=None,
                                 f1_mascaras=None):
    """matriz_*: arrays 4x4 de metricas_pertenencia (fila/columna 0 = sin clase; 1..3 = sacro, izq., der.).

    En el sistema completo la fila 0 siempre es 0 (todo fragmento real tiene clase) y la columna 0 son los
    fragmentos no encontrados: se dibuja como ultima columna. En mascaras reales la columna 0 no existe en la
    practica (la cabeza siempre da una clase) y se omite.
    """
    sistema = np.asarray(matriz_sistema)[1:, :]
    sistema = np.concatenate([sistema[:, 1:], sistema[:, :1]], axis=1)   # "no encontrado" al final
    reales = np.asarray(matriz_mascaras_reales)[1:, 1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [4, 3], "wspace": 0.12})
    fig.patch.set_facecolor(SUPERFICIE)
    sub1 = "Instancias predichas; lo no encontrado cuenta como error" + (
        f"\nF1 macro {f1_sistema:.3f}" if f1_sistema is not None else "")
    sub2 = "Solo clasificacion, sin errores de segmentacion" + (
        f"\nF1 macro {f1_mascaras:.3f}" if f1_mascaras is not None else "")
    _panel(ax1, sistema, (*ETIQUETAS, "No\nencontrado"), "Sistema completo", sub1)
    im = _panel(ax2, reales, ETIQUETAS, "Mascaras reales", sub2, con_ylabel=False)
    barra = fig.colorbar(im, ax=[ax1, ax2], fraction=0.022, pad=0.03)
    barra.set_label("Fraccion de los fragmentos de la clase real (fila)", fontsize=8.5, color=TINTA_SECUNDARIA)
    barra.ax.tick_params(labelsize=8, colors=TINTA_SECUNDARIA, length=0)
    barra.outline.set_visible(False)
    if titulo:
        fig.suptitle(titulo, fontsize=12.5, color=TINTA, x=0.02, y=1.04, ha="left", fontweight="bold")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SUPERFICIE)
    plt.close(fig)
    return path
