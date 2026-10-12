"""Pruebas de src/definiciones_notebooks.py (sin torch).

    python -m unittest tests.test_definiciones_notebooks -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.definiciones_notebooks import cargar_definiciones


def _notebook(celdas):
    return {"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                       "source": c} for c in celdas], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


class CargarDefiniciones(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "fuente.ipynb"
        celdas = [
            "import functools\n\n@functools.lru_cache(maxsize=None)\ndef cuadrado(x):\n    return x * x\n\nprint('celda de prueba')\n",
            "VALOR = 3\n\nclass Caja:\n    lado = VALOR\n",
            "def sin_decorador(x):\n    return x + 1\n",
        ]
        self.path.write_text(json.dumps(_notebook(celdas)), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_conserva_los_decoradores(self):
        g = {}
        cargar_definiciones(self.path, {"cuadrado"}, destino=g)
        self.assertEqual(g["cuadrado"](4), 16)
        self.assertTrue(hasattr(g["cuadrado"], "cache_info"), "se perdió @functools.lru_cache")

    def test_funciones_sin_decorador_y_valores(self):
        g = {}
        cargar_definiciones(self.path, {"sin_decorador", "Caja"}, valores={"VALOR"}, destino=g)
        self.assertEqual(g["sin_decorador"](1), 2)
        self.assertEqual(g["Caja"].lado, 3)

    def test_falta_una_definicion(self):
        with self.assertRaises(RuntimeError):
            cargar_definiciones(self.path, {"no_existe"}, destino={})


if __name__ == "__main__":
    unittest.main()
