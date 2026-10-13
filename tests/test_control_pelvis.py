"""Pruebas de src/control_pelvis.py y de la regla del modelo final (src/cortes_vacios.py). Sin GPU ni torch.

    python -m unittest tests.test_control_pelvis -v
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import control_pelvis as cp
from src import cortes_vacios as cv

ROOT = Path(__file__).resolve().parents[1]
SCORES_VAL = ROOT / "resultados" / "metricas" / "nms_pertenencia" / "scores_cortes_val.csv"
# Espaciado z (mm) de los MHA de val, tal como lo imprime salida_3d.ipynb al leer los headers.
SPACING_Z_VAL = {"035": 0.8, "037": 1.01, "067": 0.8, "071": 0.8, "079": 1.0}


class Racha(unittest.TestCase):
    def test_racha_mas_larga(self):
        self.assertEqual(cp.racha_mas_larga([0, 1, 1, 0, 1, 1, 1, 0]), (4, 6, 3))
        self.assertEqual(cp.racha_mas_larga([1, 1, 0, 1, 1]), (0, 1, 2))       # empate: la primera
        self.assertEqual(cp.racha_mas_larga([0, 0]), (-1, -1, 0))
        self.assertEqual(cp.racha_mas_larga([1, 1, 1]), (0, 2, 3))


class Presente(unittest.TestCase):
    def test_umbral_en_mm_no_en_cortes(self):
        s = np.r_[np.zeros(10), np.full(100, 0.9), np.zeros(10)]
        self.assertTrue(cp.pelvis_presente(s, 1.0)["presente"])          # 100 mm
        self.assertFalse(cp.pelvis_presente(s, 0.5)["presente"])         # 50 mm
        r = cp.pelvis_presente(s, 0.8)
        self.assertEqual((r["racha_cortes"], r["inicio"], r["fin"]), (100, 10, 109))
        self.assertAlmostEqual(r["racha_mm"], 80.0)
        self.assertTrue(r["presente"])                                    # 80 mm justos

    def test_det_gate_y_score_cero(self):
        s = np.r_[np.full(50, 0.19), np.full(50, 0.25)]
        self.assertEqual(cp.pelvis_presente(s, 1.0)["racha_cortes"], 50)
        self.assertFalse(cp.pelvis_presente(np.zeros(300), 1.0, det_gate=0.0)["presente"])   # sin cajas nunca hay pelvis

    def test_rachas_cortas_no_suman(self):
        s = np.tile(np.r_[np.full(30, 0.9), np.zeros(2)], 10)          # 10 rachas de 30 mm separadas
        self.assertFalse(cp.pelvis_presente(s, 1.0)["presente"])

    def test_spacing_invalido(self):
        for sp in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                cp.pelvis_presente(np.ones(5), sp)

    def test_bundle(self):
        dets = [None, {"scores": np.array([])}, {"scores": np.array([0.1, 0.7])}, {"scores": np.array([0.3])}]
        np.testing.assert_allclose(cp.scores_de_bundle({"dets": dets}), [0, 0, 0.7, 0.3])
        r = cp.pelvis_presente_bundle({"dets": dets}, 40.0)
        self.assertEqual(r["racha_cortes"], 2)
        self.assertTrue(r["presente"])

    @unittest.skipUnless(SCORES_VAL.is_file(), "faltan los resultados de val del repo")
    def test_val_real_acepta_con_pelvis_y_rechaza_tramos_sin_pelvis(self):
        """Con los scores guardados por nms_pertenencia_pipeline: los 5 volúmenes completos tienen pelvis y
        ninguno de sus tramos sin pelvis (recortados) la tiene."""
        t = pd.read_csv(SCORES_VAL, dtype={"case_id": str})
        for caso, g in t.sort_values(["case_id", "z"]).groupby("case_id"):
            sp = SPACING_Z_VAL[caso]
            s, p = g["score_max"].to_numpy(), g["tiene_hueso"].to_numpy(bool)
            self.assertTrue(cp.pelvis_presente(s, sp)["presente"], caso)
            for posicion, ini, fin in cv.tramos_sin_pelvis(p):
                r = cp.pelvis_presente(s[ini:fin], sp)
                self.assertFalse(r["presente"], f"{caso} {posicion}: racha de {r['racha_mm']:.0f} mm")


class TramosSinPelvis(unittest.TestCase):
    def test_antes_y_despues(self):
        self.assertEqual(cv.tramos_sin_pelvis([0, 0, 1, 0, 1, 0, 0, 0]), [("antes", 0, 2), ("despues", 5, 8)])
        self.assertEqual(cv.tramos_sin_pelvis([1, 1, 0]), [("despues", 2, 3)])
        self.assertEqual(cv.tramos_sin_pelvis([0, 0, 0]), [("todo", 0, 3)])
        self.assertEqual(cv.tramos_sin_pelvis([0, 1, 1, 0, 0], min_cortes=2), [("despues", 3, 5)])


class ReglaFinal(unittest.TestCase):
    ref = {"map50": 0.864, "dice": 0.944, "f1_3d": 0.667}

    def cand(self, **kw):
        d = {"map50": 0.864, "dice": 0.944, "recall_pelvis": 0.991, "f1_3d": 0.667,
             "frag_sin_pelvis_sistema": 0, "frag_sin_pelvis_modelo": 3}
        d.update(kw)
        return d

    def test_cada_condicion_puede_descalificar(self):
        for kw, clave in (({"map50": 0.85}, "topes_2d"), ({"dice": 0.93}, "topes_2d"), ({"recall_pelvis": 0.98}, "topes_2d"),
                          ({"f1_3d": 0.62}, "f1_3d_ok"), ({"frag_sin_pelvis_sistema": 1}, "sin_pelvis_ok")):
            r = cv.regla_modelo_final({"a": self.cand(**kw)}, self.ref)
            self.assertIsNone(r["ganador"], kw)
            self.assertFalse(r["detalle"]["a"][clave], kw)

    def test_un_fragmento_por_debajo_se_tolera(self):
        r = cv.regla_modelo_final({"a": self.cand(f1_3d=0.638)}, self.ref)
        self.assertEqual(r["ganador"], "a")

    def test_gana_mayor_f1_si_la_diferencia_supera_un_fragmento(self):
        r = cv.regla_modelo_final({"a": self.cand(f1_3d=0.64, frag_sin_pelvis_modelo=0),
                                   "b": self.cand(f1_3d=0.72, frag_sin_pelvis_modelo=5)}, self.ref)
        self.assertEqual(r["ganador"], "b")

    def test_empate_desempata_por_robustez_sin_control(self):
        r = cv.regla_modelo_final({"a": self.cand(f1_3d=0.70, frag_sin_pelvis_modelo=4),
                                   "b": self.cand(f1_3d=0.68, frag_sin_pelvis_modelo=1)}, self.ref)
        self.assertEqual(r["ganador"], "b")
        self.assertIn("menos fragmentos", r["motivo"])

    def test_solo_compiten_los_aceptables(self):
        r = cv.regla_modelo_final({"a": self.cand(f1_3d=0.80, frag_sin_pelvis_sistema=2),
                                   "b": self.cand(f1_3d=0.66)}, self.ref)
        self.assertEqual((r["ganador"], r["aceptables"]), ("b", ["b"]))

    def test_falta_una_clave(self):
        with self.assertRaises(KeyError):
            cv.regla_modelo_final({"a": {"map50": 0.9}}, self.ref)


if __name__ == "__main__":
    unittest.main()
