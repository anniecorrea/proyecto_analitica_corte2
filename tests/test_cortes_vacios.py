"""Pruebas de src/cortes_vacios.py. Solo NumPy/pandas: corren sin GPU y sin torch.

    python -m unittest tests.test_cortes_vacios -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import cortes_vacios as cv

ROOT = Path(__file__).resolve().parents[1]
SCORES_VAL = ROOT / "resultados" / "metricas" / "nms_pertenencia" / "scores_cortes_val.csv"
UMBRAL_JSON = ROOT / "resultados" / "metricas" / "nms_pertenencia" / "umbral_corte.json"


def volumen(z_con_pelvis, n_z=12, ids=(1,)):
    lab = np.zeros((n_z, 6, 6), dtype=np.int16)
    for z in z_con_pelvis:
        for k, i in enumerate(ids):
            lab[z, k, k] = i
    return lab


class PerfilCortes(unittest.TestCase):
    def test_distancia_y_posicion(self):
        perfil = cv.perfil_cortes(volumen([4, 5, 8], n_z=11))
        self.assertEqual(perfil["tiene_pelvis"].tolist(),
                         [False] * 4 + [True, True, False, False, True, False, False])
        self.assertEqual(perfil["distancia_a_pelvis"].tolist(), [4, 3, 2, 1, 0, 0, 1, 1, 0, 1, 2])
        self.assertEqual(perfil["posicion"].tolist(),
                         ["antes"] * 4 + ["pelvis", "pelvis", "hueco", "hueco", "pelvis", "despues", "despues"])

    def test_cuenta_cualquier_id_de_fragmento(self):
        perfil = cv.perfil_cortes(volumen([2], n_z=4, ids=(11, 21, 30)))
        self.assertEqual(int(perfil.loc[2, "voxeles_pelvis"]), 3)

    def test_volumen_sin_pelvis(self):
        perfil = cv.perfil_cortes(np.zeros((3, 4, 4), dtype=np.int16))
        self.assertFalse(perfil["tiene_pelvis"].any())
        self.assertEqual(perfil["distancia_a_pelvis"].tolist(), [-1, -1, -1])

    def test_rechaza_2d(self):
        with self.assertRaises(ValueError):
            cv.perfil_cortes(np.zeros((4, 4)))


class ManifiestoNegativos(unittest.TestCase):
    def test_solo_cortes_sin_un_voxel_de_pelvis(self):
        casos = pd.DataFrame({"case_id": ["002", "001"], "split": ["val", "train"]})
        vols = {"001": volumen([1, 2], n_z=4), "002": volumen([0], n_z=3)}
        neg = cv.manifiesto_negativos(casos, vols.__getitem__)
        self.assertEqual(list(neg.columns), cv.COLUMNAS_NEGATIVOS)
        self.assertEqual(list(zip(neg["case_id"], neg["slice_index"])), [("001", 0), ("001", 3), ("002", 1), ("002", 2)])
        self.assertEqual(neg["split"].tolist(), ["train", "train", "val", "val"])

    def test_case_id_repetido(self):
        casos = pd.DataFrame({"case_id": ["001", "001"], "split": ["train", "val"]})
        with self.assertRaises(ValueError):
            cv.manifiesto_negativos(casos, lambda c: volumen([0]))


class Submuestreo(unittest.TestCase):
    def setUp(self):
        self.neg = pd.DataFrame({"split": "train", "case_id": ["001"] * 10 + ["002"] * 3,
                                 "slice_index": list(range(10)) + [0, 1, 2],
                                 "distancia_a_pelvis": 1, "posicion": "antes"})

    def test_reproducible_y_por_caso(self):
        a = cv.submuestrear_negativos(self.neg, 0.5, seed=7)
        b = cv.submuestrear_negativos(self.neg, 0.5, seed=7)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.groupby("case_id").size().to_dict(), {"001": 5, "002": 2})   # ceil(1.5) = 2

    def test_fraccion_uno_devuelve_todo(self):
        self.assertEqual(len(cv.submuestrear_negativos(self.neg, 1.0)), len(self.neg))

    def test_fraccion_invalida(self):
        for f in (0.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                cv.submuestrear_negativos(self.neg, f)


class Metricas(unittest.TestCase):
    def test_regla_procesa(self):
        # score 0 = sin cajas: nunca se procesa, ni con umbral 0
        np.testing.assert_array_equal(cv.procesa([0.0, 0.2, 0.5], 0.0), [False, True, True])
        np.testing.assert_array_equal(cv.procesa([0.0, 0.2, 0.5], 0.5), [False, False, True])

    def test_metricas_a_mano(self):
        scores = [0.9, 0.8, 0.3, 0.0, 0.6, 0.5, 0.0, 0.1]
        pelvis = [1, 1, 1, 1, 0, 0, 0, 0]
        m = cv.metricas_cortes_vacios(scores, pelvis, 0.5)
        self.assertAlmostEqual(m["recall_pelvis"], 0.5)
        self.assertAlmostEqual(m["fpr_vacios"], 0.5)
        self.assertEqual(m["vacios_que_pasan"], 2)
        self.assertAlmostEqual(m["vacios_con_caja"], 0.75)

    def test_calibrar_elige_el_mayor_umbral_con_recall(self):
        scores = np.r_[np.linspace(0.30, 1.0, 100), np.linspace(0.0, 0.9, 50)]
        pelvis = np.r_[np.ones(100, bool), np.zeros(50, bool)]
        u, tabla = cv.calibrar_umbral(scores, pelvis, recall_min=0.99)
        self.assertGreaterEqual(tabla.set_index("umbral").loc[u, "recall_hueso"], 0.99)
        siguiente = tabla[tabla["umbral"] > u + 1e-9]
        if len(siguiente):
            self.assertLess(siguiente.iloc[0]["recall_hueso"], 0.99)

    def test_longitudes_distintas(self):
        with self.assertRaises(ValueError):
            cv.metricas_cortes_vacios([0.1, 0.2], [1], 0.5)

    @unittest.skipUnless(SCORES_VAL.is_file() and UMBRAL_JSON.is_file(), "faltan los resultados de val del repo")
    def test_reproduce_los_numeros_reales_de_val(self):
        """Con los scores guardados por nms_pertenencia_pipeline debe salir lo mismo que su umbral_corte.json."""
        t = pd.read_csv(SCORES_VAL, dtype={"case_id": str})
        guardado = json.loads(UMBRAL_JSON.read_text(encoding="utf-8"))
        u, tabla = cv.calibrar_umbral(t["score_max"], t["tiene_hueso"], recall_min=guardado["recall_min_hueso"])
        self.assertAlmostEqual(u, guardado["umbral_corte"])
        fila = tabla.set_index("umbral").loc[u]
        self.assertAlmostEqual(fila["recall_hueso"], guardado["recall_hueso"], places=9)
        self.assertAlmostEqual(fila["rechazo_vacios"], guardado["rechazo_vacios"], places=9)
        r = cv.resumen_cortes_vacios(t["score_max"], t["tiene_hueso"], umbral_fijo=guardado["umbral_corte"])
        vacios = t.loc[~t["tiene_hueso"].astype(bool), "score_max"]
        self.assertEqual((r["n_pelvis"], r["n_vacios"]), (guardado["cortes_con_hueso"], guardado["cortes"] - guardado["cortes_con_hueso"]))
        self.assertAlmostEqual(r["fpr_vacios_fijo"], 1 - guardado["rechazo_vacios"])
        self.assertAlmostEqual(r["fpr_vacios_fijo"], ((vacios > 0) & (vacios >= guardado["umbral_corte"])).mean())
        self.assertAlmostEqual(r["vacios_con_caja"], (vacios > 0).mean())


class EquivalenciaConElPipeline(unittest.TestCase):
    """calibrar_umbral debe dar lo mismo que pipeline_pengwin.calibrar_umbral_corte (no se duplica una regla distinta)."""
    def test_misma_tabla(self):
        try:
            from src.pipeline_pengwin import calibrar_umbral_corte
        except ImportError as exc:                      # sin torch instalado
            self.skipTest(f"pipeline_pengwin no se puede importar: {exc}")
        rng = np.random.default_rng(0)
        scores = np.where(rng.random(400) < 0.2, 0.0, rng.random(400))
        pelvis = rng.random(400) < 0.7
        u1, t1 = cv.calibrar_umbral(scores, pelvis)
        u2, t2 = calibrar_umbral_corte(scores, pelvis)
        self.assertEqual(u1, u2)
        pd.testing.assert_frame_equal(t1, t2)


class CriterioSeleccion(unittest.TestCase):
    base = {"val_det_map50": 0.889, "val_seg_dice_pred": 0.948, "val_fpr_vacios_cal": 0.48,
            "val_recall_pelvis_cal": 0.991}

    def m(self, **kw):
        d = dict(self.base)
        d.update({f"val_{k}": v for k, v in kw.items()})
        return d

    def test_cumple_y_puntua_por_fpr(self):
        ok, p, det = cv.criterio_seleccion(self.m(fpr_vacios_cal=0.20, det_map50=0.882), self.base)
        self.assertTrue(ok)
        self.assertAlmostEqual(p, 0.80)
        self.assertAlmostEqual(det["mejora_fpr"], 0.28)

    def test_caida_de_map_o_dice_o_recall_no_cumple(self):
        for kw in ({"det_map50": 0.870}, {"seg_dice_pred": 0.930}, {"recall_pelvis_cal": 0.98}):
            ok, p, _ = cv.criterio_seleccion(self.m(fpr_vacios_cal=0.05, **kw), self.base)
            self.assertFalse(ok, kw)
            self.assertLess(p, 0)

    def test_una_epoca_que_cumple_siempre_gana(self):
        _, p_mala, _ = cv.criterio_seleccion(self.m(fpr_vacios_cal=0.0, det_map50=0.85), self.base)
        _, p_buena, _ = cv.criterio_seleccion(self.m(fpr_vacios_cal=0.99), self.base)
        self.assertGreater(p_buena, p_mala)

    def test_falta_una_metrica(self):
        with self.assertRaises(KeyError):
            cv.criterio_seleccion({"val_det_map50": 0.9}, self.base)


class TramoPrincipal(unittest.TestCase):
    def test_conserva_el_grupo_mas_largo(self):
        a = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1], bool)
        np.testing.assert_array_equal(cv.tramo_principal(a, 0), [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(cv.tramo_principal(a, 1), [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
        np.testing.assert_array_equal(cv.tramo_principal(a, 3), a)

    def test_nunca_activa_cortes_nuevos_y_vacio(self):
        a = np.array([1, 0, 1], bool)
        self.assertFalse((cv.tramo_principal(a, 5) & ~a).any())
        self.assertFalse(cv.tramo_principal(np.zeros(4, bool), 2).any())

    def test_empate_gana_el_primero(self):
        np.testing.assert_array_equal(cv.tramo_principal(np.array([1, 0, 0, 1], bool), 0), [1, 0, 0, 0])

    @unittest.skipUnless(SCORES_VAL.is_file(), "faltan los resultados de val del repo")
    def test_tabla_con_datos_reales(self):
        t = pd.read_csv(SCORES_VAL, dtype={"case_id": str})
        u = json.loads(UMBRAL_JSON.read_text(encoding="utf-8"))["umbral_corte"] if UMBRAL_JSON.is_file() else 0.5
        vacios = t.loc[~t["tiene_hueso"].astype(bool), "score_max"]
        pasan = int(((vacios > 0) & (vacios >= u)).sum())
        tabla = cv.evaluar_tramo_principal(t, u, huecos=(0, 2))
        self.assertEqual(tabla["vacios_que_pasan_antes"].tolist(), [pasan, pasan])
        self.assertTrue((tabla["vacios_que_pasan_despues"] <= pasan).all())
        self.assertEqual(int(tabla.loc[0, "n_cortes_pelvis"]), int(t["tiene_hueso"].astype(bool).sum()))


if __name__ == "__main__":
    unittest.main()
