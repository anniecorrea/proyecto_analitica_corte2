import unittest

import numpy as np

from src.plausibilidad_pelvis import (COXAL_DER, COXAL_IZQ, SACRO, escalar, evaluar, evaluar_niveles,
                                      factor_tolerancia, intervalo_tolerancia, medidas_pelvis,
                                      regiones_desde_etiqueta, regiones_desde_instancias)


def pelvis_sintetica():
    """Tres bloques en orden lateral humano (x crece hacia la izquierda del paciente): coxal der., sacro, coxal izq."""
    v = np.zeros((20, 10, 60), dtype=np.uint8)
    v[2:18, 2:8, 0:20] = COXAL_DER
    v[4:16, 3:7, 24:36] = SACRO
    v[2:18, 2:8, 40:60] = COXAL_IZQ
    return v


SPACING = (5.0, 15.0, 10.0)      # x, y, z en mm: 60 x 5 = 300 mm de ancho, 6 x 15 = 90 mm, 16 x 10 = 160 mm


class MedidasTests(unittest.TestCase):
    def test_tamano_volumen_y_proporciones(self):
        m = medidas_pelvis(pelvis_sintetica(), SPACING)
        self.assertAlmostEqual(m["ancho_mm"], 300.0, delta=10.0)
        self.assertAlmostEqual(m["profundidad_mm"], 90.0, delta=15.0)
        self.assertAlmostEqual(m["alto_mm"], 160.0, delta=10.0)
        vox = 5.0 * 15.0 * 10.0 / 1000.0
        self.assertAlmostEqual(m["volumen_cm3"], (2 * 16 * 6 * 20 + 12 * 4 * 12) * vox)
        self.assertAlmostEqual(m["simetria_coxales"], 1.0)
        self.assertTrue(m["orden_lateral_ok"])
        self.assertEqual(m["faltan"], [])

    def test_un_voxel_suelto_lejos_no_infla_el_tamano(self):
        v = np.zeros((20, 10, 400), dtype=np.uint8)
        v[:, :, 100:160] = pelvis_sintetica()
        con_suelto = v.copy()
        con_suelto[10, 5, 399] = COXAL_IZQ                      # un vóxel a 240 vóxeles (1200 mm) de la pelvis
        self.assertAlmostEqual(medidas_pelvis(con_suelto, SPACING)["ancho_mm"],
                               medidas_pelvis(v, SPACING)["ancho_mm"], delta=10.0)

    def test_lados_invertidos_y_hueso_faltante(self):
        self.assertFalse(medidas_pelvis(pelvis_sintetica()[:, :, ::-1], SPACING)["orden_lateral_ok"])
        sin_sacro = pelvis_sintetica()
        sin_sacro[sin_sacro == SACRO] = 0
        m = medidas_pelvis(sin_sacro, SPACING)
        self.assertEqual(m["faltan"], ["sacro"])
        self.assertFalse(m["orden_lateral_ok"])

    def test_conversion_de_etiquetas_e_instancias(self):
        etiqueta = np.array([0, 1, 10, 11, 20, 21, 30])
        np.testing.assert_array_equal(regiones_desde_etiqueta(etiqueta), [0, 1, 1, 2, 2, 3, 3])
        inst = np.array([0, 1, 2, 3])
        regiones = regiones_desde_instancias(inst, {1: "coxal_derecho", 2: "sacro", 3: "coxal_izquierdo"})
        np.testing.assert_array_equal(regiones, [0, COXAL_DER, SACRO, COXAL_IZQ])


class EvaluarTests(unittest.TestCase):
    def setUp(self):
        self.m = medidas_pelvis(pelvis_sintetica(), SPACING)
        # referencia hecha a la medida de la pelvis sintética: esa pelvis es "típica"
        self.ref = {"rangos": {k: {"min": self.m[k] * 0.9, "max": self.m[k] * 1.1}
                               for k in ("ancho_mm", "profundidad_mm", "alto_mm", "volumen_cm3", "fraccion_sacro",
                                         "simetria_coxales")}}

    def test_pelvis_tipica_no_avisa(self):
        self.assertEqual(evaluar(self.m, self.ref), [])

    def test_otro_tamano_avisa_y_dice_por_que(self):
        for factor in (0.5, 1.6):
            with self.subTest(factor=factor):
                motivos = evaluar(escalar(self.m, factor), self.ref)
                self.assertTrue(any("ancho de la pelvis" in t for t in motivos))
                self.assertTrue(any("volumen de hueso" in t for t in motivos))

    def test_lados_invertidos_avisa(self):
        motivos = evaluar(self.m | {"orden_lateral_ok": False}, self.ref)
        self.assertTrue(any("orden esperado" in t for t in motivos))

    def test_hueso_faltante_avisa(self):
        motivos = evaluar(self.m | {"faltan": ["sacro"], "orden_lateral_ok": False}, self.ref)
        self.assertTrue(any("no se encontró: sacro" in t for t in motivos))


class NivelesTests(unittest.TestCase):
    """Nota si la medida sale del rango de entrenamiento; advertencia solo si sale del intervalo de tolerancia."""

    def setUp(self):
        self.m = medidas_pelvis(pelvis_sintetica(), SPACING)
        self.ref = {"rangos": {k: {"min": self.m[k] * 0.95, "max": self.m[k] * 1.05,
                                   "tolerancia_min": self.m[k] * 0.8, "tolerancia_max": self.m[k] * 1.25}
                               for k in ("ancho_mm", "profundidad_mm", "alto_mm", "volumen_cm3", "fraccion_sacro")}}

    def test_dentro_del_rango_no_dice_nada(self):
        self.assertEqual(evaluar_niveles(self.m, self.ref), {"avisos": [], "notas": []})

    def test_fuera_del_rango_pero_dentro_de_la_tolerancia_solo_es_nota(self):
        m = self.m | {"ancho_mm": self.m["ancho_mm"] * 1.1}
        niveles = evaluar_niveles(m, self.ref)
        self.assertEqual(niveles["avisos"], [])
        self.assertTrue(any("ancho de la pelvis" in t for t in niveles["notas"]))

    def test_fuera_de_la_tolerancia_es_advertencia(self):
        m = self.m | {"ancho_mm": self.m["ancho_mm"] * 1.4}
        niveles = evaluar_niveles(m, self.ref)
        self.assertTrue(any("ancho de la pelvis" in t for t in niveles["avisos"]))
        self.assertFalse(any("ancho de la pelvis" in t for t in niveles["notas"]))

    def test_factor_de_tolerancia_coincide_con_tablas(self):
        # n = 75, 99 % de cobertura y 95 % de confianza: k ≈ 3.00 (tablas de intervalos de tolerancia normales)
        self.assertAlmostEqual(factor_tolerancia(75, 0.99, 0.95), 3.00, delta=0.03)
        self.assertGreater(factor_tolerancia(75, 0.999, 0.99), factor_tolerancia(75, 0.99, 0.95))
        lo, hi = intervalo_tolerancia([100.0, 110.0, 120.0, 105.0, 115.0])
        self.assertLess(lo, 100.0)
        self.assertGreater(hi, 120.0)


if __name__ == "__main__":
    unittest.main()
