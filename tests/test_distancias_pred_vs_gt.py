import unittest

import numpy as np

from src.distancias_pred_vs_gt import comparar_distancias


class CompararDistanciasTests(unittest.TestCase):
    spacing_xyz = (1.0, 1.0, 1.0)

    def setUp(self):
        # Referencia: sacro con principal (1), un conminuto separado 3 vóxeles (2) y uno pegado al principal (3)
        self.gt = np.zeros((6, 6, 20), dtype=np.int16)
        self.gt[:, :, 0:6] = 1
        self.gt[:, :, 9:12] = 2
        self.gt[:, :, 6:8] = 3

    def fila(self, tabla, gt_id):
        return tabla.loc[tabla["gt_id"] == gt_id].iloc[0]

    def test_prediccion_perfecta_separa_y_mide_igual(self):
        tabla = comparar_distancias(self.gt.copy(), {1: "sacro", 2: "sacro", 3: "sacro"}, self.gt, self.spacing_xyz)
        for gt_id in (2, 3):
            fila = self.fila(tabla, gt_id)
            self.assertEqual(fila["estado"], "separado")
            self.assertAlmostEqual(fila["error_abs_mm"], 0.0)
        self.assertAlmostEqual(self.fila(tabla, 2)["gt_distancia_mm"], 3.0)    # hueco de 3 vóxeles de 1 mm (x = 6, 7, 8)
        self.assertTrue(self.fila(tabla, 1)["principal_ok"])

    def test_conminuto_fundido_con_el_principal_cuenta_como_fusionado_a_cero_mm(self):
        pred = self.gt.copy()
        pred[pred == 3] = 1                                    # el modelo pega el trozo en contacto al principal
        pred[pred == 2] = 1                                    # y también el separado
        tabla = comparar_distancias(pred, {1: "sacro"}, self.gt, self.spacing_xyz)
        for gt_id in (2, 3):
            fila = self.fila(tabla, gt_id)
            self.assertEqual(fila["estado"], "fusionado")
            self.assertEqual(fila["pred_distancia_mm"], 0.0)
        self.assertAlmostEqual(self.fila(tabla, 2)["error_abs_mm"], 3.0)        # se pierde la distancia real entera

    def test_conminuto_ausente_en_la_prediccion_es_omitido_sin_distancia(self):
        pred = self.gt.copy()
        pred[pred == 2] = 0
        tabla = comparar_distancias(pred, {1: "sacro", 3: "sacro"}, self.gt, self.spacing_xyz)
        fila = self.fila(tabla, 2)
        self.assertEqual(fila["estado"], "omitido")
        self.assertTrue(np.isnan(fila["pred_distancia_mm"]))

    def test_geometria_distinta_falla(self):
        with self.assertRaises(ValueError):
            comparar_distancias(self.gt[:-1], {1: "sacro"}, self.gt, self.spacing_xyz)


if __name__ == "__main__":
    unittest.main()
