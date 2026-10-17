"""Pruebas de la figura de confusion de la pertenencia (sin PyTorch)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.figuras_pertenencia import _por_fila, figura_confusion_pertenencia  # noqa: E402

SISTEMA = np.array([[0, 0, 0, 0], [335, 721, 6, 0], [263, 20, 1170, 24], [257, 30, 54, 1065]])
REALES = np.array([[0, 0, 0, 0], [0, 1045, 7, 10], [0, 77, 1374, 26], [0, 43, 53, 1310]])


class FiguraConfusion(unittest.TestCase):
    def test_filas_suman_uno(self):
        frac = _por_fila(SISTEMA[1:])
        np.testing.assert_allclose(frac.sum(axis=1), 1.0)

    def test_clase_sin_fragmentos_no_da_nan(self):
        frac = _por_fila(np.array([[0, 0, 0], [2, 1, 1]]))
        self.assertFalse(np.isnan(frac).any())
        np.testing.assert_allclose(frac[0], 0.0)

    def test_genera_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = figura_confusion_pertenencia(SISTEMA, REALES, Path(tmp) / "sub" / "conf.png", "prueba", 0.8, 0.9)
            self.assertTrue(ruta.is_file())
            self.assertGreater(ruta.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
