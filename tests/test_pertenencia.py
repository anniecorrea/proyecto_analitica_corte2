"""Pruebas de la cabeza de pertenencia (pertenencia.py). Autor: Fabián.

Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_pertenencia -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pertenencia import (PertenenciaHead, cargar_submodulo, clase_de_id, clasificar_instancias,  # noqa: E402
                         entrenar_pertenencia, instancias_de_mascara, metricas_pertenencia, pesos_de_mascaras,
                         perdida_pertenencia, pesos_por_clase, precalcular_cache_pertenencia, predecir_cache,
                         rasgos_geometricos, regla_caja_detectada, regla_mayoria_semantica, targets_pertenencia)


def mascara_tres_huesos(h=256, w=256):
    """Sacro al centro (ID 1), coxal izquierdo a un lado (ID 12) y coxal derecho al otro (ID 25)."""
    m = np.zeros((h, w), dtype=np.int64)
    m[100:140, 110:146] = 1
    m[120:180, 30:90] = 12
    m[120:180, 166:226] = 25
    return m


class TargetsYEtiquetas(unittest.TestCase):
    def test_clase_de_id(self):
        self.assertEqual([clase_de_id(i) for i in (0, 1, 10, 11, 20, 21, 30, 31)], [0, 1, 1, 2, 2, 3, 3, 0])

    def test_targets_de_un_lote(self):
        vacia = np.zeros((256, 256), dtype=np.int64)
        masks, b_idx, labels = targets_pertenencia([mascara_tres_huesos(), vacia, mascara_tres_huesos()])
        self.assertEqual(masks.shape, (6, 256, 256))
        self.assertEqual(b_idx.tolist(), [0, 0, 0, 2, 2, 2])           # la imagen 1 no aporta instancias
        self.assertEqual(labels.tolist(), [0, 1, 2, 0, 1, 2])          # 1 -> 0, 12 -> 1, 25 -> 2

    def test_lote_sin_instancias(self):
        masks, b_idx, labels = targets_pertenencia([np.zeros((256, 256), dtype=np.int64)])
        self.assertEqual(masks.shape, (0, 256, 256))
        self.assertEqual(b_idx.numel() + labels.numel(), 0)

    def test_id_fuera_de_rango_falla(self):
        m = np.zeros((8, 8), dtype=np.int64)
        m[0, 0] = 45
        with self.assertRaises(ValueError):
            targets_pertenencia([m])

    def test_acepta_uint16_del_decoder(self):
        masks, ids = instancias_de_mascara(mascara_tres_huesos().astype(np.uint16))
        self.assertEqual(ids.tolist(), [1, 12, 25])

    def test_pesos_por_clase_favorecen_la_minoritaria(self):
        w = pesos_por_clase(torch.tensor([0, 0, 0, 0, 1, 1, 2]))
        self.assertGreater(w[2].item(), w[0].item())
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)


class Geometria(unittest.TestCase):
    def test_posicion_relativa_al_centro_del_hueso(self):
        masks, b_idx, _ = targets_pertenencia([mascara_tres_huesos()])
        g = rasgos_geometricos(masks, b_idx)
        self.assertEqual(g.shape, (3, 6))
        self.assertTrue(torch.isfinite(g).all())
        cx_rel = g[:, 2]
        self.assertLess(cx_rel[1].item(), 0)                           # ID 12 está a la izquierda del centro
        self.assertGreater(cx_rel[2].item(), 0)                        # ID 25 está a la derecha
        self.assertLess(abs(cx_rel[0].item()), 0.05)                   # el sacro está casi en el centro


class Cabeza(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.head = PertenenciaHead()
        self.fmap = torch.randn(2, 256, 16, 16)

    def test_forma_de_salida(self):
        masks, b_idx, _ = targets_pertenencia([mascara_tres_huesos(), mascara_tres_huesos()])
        self.assertEqual(self.head(self.fmap, masks, b_idx).shape, (6, 3))

    def test_cero_instancias_no_rompe_ni_la_perdida(self):
        logits = self.head(self.fmap, torch.zeros(0, 256, 256, dtype=torch.bool), torch.zeros(0, dtype=torch.long))
        self.assertEqual(logits.shape, (0, 3))
        loss = perdida_pertenencia(logits, torch.zeros(0, dtype=torch.long))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()                                                # conectado al grafo: no lanza error

    def test_fragmento_de_un_pixel(self):
        m = np.zeros((256, 256), dtype=np.int64)
        m[50, 200] = 23
        masks, b_idx, _ = targets_pertenencia([m])
        logits = self.head(self.fmap[:1], masks, b_idx)
        self.assertTrue(torch.isfinite(logits).all())

    def test_gradiente_llega_al_mapa_compartido(self):
        fmap = self.fmap.clone().requires_grad_(True)
        masks, b_idx, labels = targets_pertenencia([mascara_tres_huesos(), mascara_tres_huesos()])
        perdida_pertenencia(self.head(fmap, masks, b_idx), labels).backward()
        self.assertIsNotNone(fmap.grad)
        self.assertGreater(fmap.grad.abs().sum().item(), 0)            # el backbone también aprende de esta cabeza

    def test_batch_idx_invalido_falla(self):
        masks, _, _ = targets_pertenencia([mascara_tres_huesos()])
        with self.assertRaises(ValueError):
            self.head(self.fmap, masks, torch.tensor([0, 1, 5]))

    def test_overfit_corto_aprende_izquierda_y_derecha(self):
        """Prueba técnica (no de desempeño): con mapas aleatorios la única pista es la posición.
        Si la cabeza no puede memorizar esto, algo está mal conectado."""
        torch.manual_seed(1)
        rng = np.random.default_rng(1)
        mascaras = []
        for _ in range(8):
            m = np.zeros((256, 256), dtype=np.int64)
            dx = int(rng.integers(-15, 15))
            m[100:130, 113 + dx:143 + dx] = 2                          # sacro
            m[110:170, 30 + dx:80 + dx] = 14                           # coxal izquierdo
            m[110:170, 176 + dx:226 + dx] = 27                         # coxal derecho
            mascaras.append(m)
        fmap = torch.randn(8, 256, 16, 16)
        masks, b_idx, labels = targets_pertenencia(mascaras)
        head = PertenenciaHead(dropout=0.0)
        opt = torch.optim.Adam(head.parameters(), lr=3e-3)
        for _ in range(300):
            opt.zero_grad()
            loss = perdida_pertenencia(head(fmap, masks, b_idx), labels)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = head(fmap, masks, b_idx).argmax(dim=1)
        self.assertEqual(pred.tolist(), labels.tolist())
        self.assertLess(loss.item(), 0.1)

    @unittest.skipUnless(torch.cuda.is_available(), "sin GPU")
    def test_amp_en_gpu(self):
        head = self.head.cuda()
        masks, b_idx, labels = targets_pertenencia([mascara_tres_huesos()])
        with torch.autocast("cuda", dtype=torch.float16):
            logits = head(self.fmap[:1].cuda(), masks.cuda(), b_idx.cuda())
            loss = perdida_pertenencia(logits, labels.cuda())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


class Inferencia(unittest.TestCase):
    def test_clasificar_instancias_predichas(self):
        head = PertenenciaHead().eval()
        fmap = torch.randn(3, 256, 16, 16)
        pred = [mascara_tres_huesos().astype(np.uint16), np.zeros((256, 256), dtype=np.uint16),
                (mascara_tres_huesos() > 0).astype(np.uint16) * 7]
        res = clasificar_instancias(head, fmap, pred)
        self.assertEqual(sorted(res[0]), [1, 12, 25])
        self.assertEqual(res[1], {})                                   # imagen sin instancias
        self.assertEqual(sorted(res[2]), [7])
        for p in res[0].values():
            self.assertAlmostEqual(float(p.sum()), 1.0, places=5)

    def test_reglas_de_comparacion(self):
        inst = mascara_tres_huesos()
        sem = np.vectorize(clase_de_id)(inst)
        self.assertEqual(regla_mayoria_semantica(inst, sem), {1: 1, 12: 2, 25: 3})
        cajas = np.array([[100, 90, 150, 150], [20, 110, 100, 190]], dtype=np.float32)
        self.assertEqual(regla_caja_detectada(inst, cajas, np.array([1, 2])), {1: 1, 12: 2, 25: 0})


class Metricas(unittest.TestCase):
    def test_prediccion_perfecta(self):
        y = np.array([1, 2, 3, 1, 2, 3])
        probs = np.eye(3)[y - 1] * 0.9 + 0.1 / 3
        m = metricas_pertenencia(y, probs=probs)
        self.assertEqual(m["f1_macro"], 1.0)
        self.assertEqual(m["auc_macro_ovr"], 1.0)

    def test_sin_clase_cuenta_como_error(self):
        m = metricas_pertenencia(np.array([1, 2, 3]), y_pred=np.array([1, 2, 0]))
        self.assertLess(m["f1_macro"], 1.0)
        self.assertEqual(m["matriz_confusion"][3, 0], 1)               # el coxal derecho quedó sin clase

    def test_vacio(self):
        self.assertTrue(np.isnan(metricas_pertenencia(np.array([]), y_pred=np.array([]))["f1_macro"]))


class EtapaD(unittest.TestCase):
    """Entrenamiento de la cabeza sobre un tronco congelado con mapas precalculados."""

    def setUp(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        self.tronco = torch.nn.Conv2d(1, 256, kernel_size=16, stride=16).eval()     # tronco falso: [B,1,256,256] -> [B,256,16,16]
        self.dataset = []
        for k in range(12):
            m = np.zeros((256, 256), dtype=np.int64)
            dx = int(rng.integers(-15, 15))
            m[100:130, 113 + dx:143 + dx] = 3
            m[110:170, 30 + dx:80 + dx] = 11 + k % 5
            if k % 3:                                                  # no todos los cortes tienen coxal derecho
                m[110:170, 176 + dx:226 + dx] = 24
            img = torch.from_numpy((m > 0).astype(np.float32))[None] + 0.1 * torch.randn(1, 256, 256)
            self.dataset.append((img, {"instance_mask": torch.from_numpy(m), "case_id": f"{k // 4:03d}", "slice_index": k}))

    def test_cache_tiene_formas_y_meta_alineadas(self):
        cache = precalcular_cache_pertenencia(self.tronco, self.dataset, range(12), batch_size=5, progreso=False)
        K = cache["label"].shape[0]
        self.assertEqual(cache["f4"].shape, (12, 256, 16, 16))
        self.assertEqual(cache["f4"].dtype, torch.float16)
        self.assertEqual(K, 12 + 12 + 8)                               # sacro + coxal izq. en todos, derecho en 8
        self.assertEqual(cache["pesos"].shape, (K, 16, 16))
        self.assertEqual(cache["geom"].shape, (K, 6))
        self.assertEqual(len(cache["meta"]), K)
        self.assertEqual(int(cache["corte"].max()), 11)
        # la etiqueta corresponde al ID real guardado en meta
        self.assertEqual((cache["label"] + 1).tolist(), [clase_de_id(i) for i in cache["meta"]["gt_id"]])

    def test_forward_desde_pesos_igual_a_forward(self):
        head = PertenenciaHead().eval()
        fmap = torch.randn(2, 256, 16, 16)
        masks, b_idx, _ = targets_pertenencia([mascara_tres_huesos(), mascara_tres_huesos()])
        a = head(fmap, masks, b_idx)
        b = head.forward_desde_pesos(fmap, pesos_de_mascaras(masks, (16, 16)), rasgos_geometricos(masks, b_idx), b_idx)
        torch.testing.assert_close(a, b)

    def test_entrenamiento_aprende_y_selecciona_con_val(self):
        train = precalcular_cache_pertenencia(self.tronco, self.dataset, range(0, 8), progreso=False)
        val = precalcular_cache_pertenencia(self.tronco, self.dataset, range(8, 12), progreso=False)
        head, historial, mejor = entrenar_pertenencia(PertenenciaHead(dropout=0.0), train, val, epochs=150,
                                                      lr=3e-3, batch_inst=16, paciencia=150)
        self.assertGreaterEqual(mejor["val_f1_macro"], 0.99)
        self.assertEqual(int(historial["es_mejor"].sum()) >= 1, True)
        self.assertFalse(head.training)                                # queda en eval con los mejores pesos
        probs = predecir_cache(head, val)
        self.assertEqual(probs.shape, (val["label"].shape[0], 3))
        self.assertEqual((probs.argmax(1) == val["label"]).float().mean().item(), 1.0)

    def test_cargar_submodulo_es_estricto(self):
        destino = torch.nn.Linear(3, 2)
        origen = {"backbone.weight": torch.ones(2, 3), "backbone.bias": torch.zeros(2), "det_head.x": torch.ones(1)}
        cargar_submodulo(destino, origen, "backbone.")
        self.assertTrue(torch.equal(destino.weight, torch.ones(2, 3)))
        with self.assertRaises(KeyError):
            cargar_submodulo(destino, origen, "cbam.")
        with self.assertRaises(RuntimeError):
            cargar_submodulo(destino, {"backbone.weight": torch.ones(2, 3)}, "backbone.")   # falta el bias


if __name__ == "__main__":
    unittest.main()
