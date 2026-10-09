"""Pruebas del NMS propio por clase (deteccion_grid.nms_por_clase). Autor: Fabián.

Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_nms -v
"""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # para importar los módulos de la raíz
from deteccion_grid import NMS_CONFIG, decode_predictions, nms_por_clase, postprocess  # noqa: E402

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def cajas(*filas):
    return torch.tensor(filas, dtype=torch.float32).reshape(-1, 4)


class NMSReglasBasicas(unittest.TestCase):
    def test_entrada_vacia_devuelve_tensor_vacio(self):
        keep = nms_por_clase(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long))
        self.assertEqual(keep.shape, (0,))
        self.assertEqual(keep.dtype, torch.long)

    def test_duplicado_de_la_misma_clase_se_elimina(self):
        b = cajas([10, 10, 50, 50], [12, 12, 52, 52])                 # IoU ~ 0.82
        keep = nms_por_clase(b, torch.tensor([0.6, 0.9]), torch.tensor([1, 1]), iou_thresh=0.5)
        self.assertEqual(keep.tolist(), [1])                           # sobrevive la de mayor score

    def test_clases_distintas_no_se_suprimen(self):
        b = cajas([10, 10, 50, 50], [10, 10, 50, 50])                 # misma caja, IoU = 1
        keep = nms_por_clase(b, torch.tensor([0.6, 0.9]), torch.tensor([1, 2]), iou_thresh=0.5)
        self.assertEqual(keep.tolist(), [1, 0])                        # ambas, ordenadas por score

    def test_cajas_separadas_de_la_misma_clase(self):
        b = cajas([0, 0, 10, 10], [100, 100, 120, 120])
        s, l = torch.tensor([0.7, 0.8]), torch.tensor([3, 3])
        self.assertEqual(nms_por_clase(b, s, l, max_det_per_class=None).tolist(), [1, 0])
        self.assertEqual(nms_por_clase(b, s, l, max_det_per_class=1).tolist(), [1])

    def test_iou_igual_al_umbral_se_conserva(self):
        b = cajas([0, 0, 10, 10], [0, 0, 10, 5])                      # IoU = 50 / 100 = 0.5
        s, l = torch.tensor([0.9, 0.8]), torch.tensor([1, 1])
        self.assertEqual(nms_por_clase(b, s, l, iou_thresh=0.5).tolist(), [0, 1])
        self.assertEqual(nms_por_clase(b, s, l, iou_thresh=0.49).tolist(), [0])

    def test_cadena_voraz(self):
        # A solapa con B (IoU 0.54) y B con C (0.54), pero A con C solo 0.25.
        # B se elimina por A; C sobrevive porque B ya no está para eliminarla.
        b = cajas([0, 0, 10, 10], [3, 0, 13, 10], [6, 0, 16, 10])
        keep = nms_por_clase(b, torch.tensor([0.9, 0.8, 0.7]), torch.tensor([2, 2, 2]),
                             iou_thresh=0.5, max_det_per_class=None)
        self.assertEqual(keep.tolist(), [0, 2])

    def test_empate_de_score_gana_el_indice_menor(self):
        b = cajas([10, 10, 50, 50], [10, 10, 50, 50])
        keep = nms_por_clase(b, torch.tensor([0.5, 0.5]), torch.tensor([1, 1]))
        self.assertEqual(keep.tolist(), [0])


class NMSEntradasProblematicas(unittest.TestCase):
    def test_score_thresh_puede_dejar_todo_vacio(self):
        b = cajas([0, 0, 10, 10], [20, 20, 30, 30])
        keep = nms_por_clase(b, torch.tensor([0.1, 0.2]), torch.tensor([1, 2]), score_thresh=0.5)
        self.assertEqual(keep.numel(), 0)

    def test_descarta_degeneradas_y_no_finitas(self):
        b = cajas([5, 5, 5, 10],                                       # ancho 0
                  [0, 0, 10, 10],                                      # score NaN
                  [0, 0, float("nan"), 10],                            # coordenada NaN
                  [40, 40, 60, 60])                                    # única válida
        s = torch.tensor([0.9, float("nan"), 0.8, 0.3])
        keep = nms_por_clase(b, s, torch.tensor([1, 1, 1, 1]), max_det_per_class=None)
        self.assertEqual(keep.tolist(), [3])

    def test_float16_da_el_mismo_resultado(self):
        b = cajas([10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 140, 140])
        s, l = torch.tensor([0.6, 0.9, 0.5]), torch.tensor([1, 1, 1])
        ref = nms_por_clase(b, s, l, max_det_per_class=None)
        half = nms_por_clase(b.half(), s.half(), l, max_det_per_class=None)
        self.assertEqual(ref.tolist(), half.tolist())

    def test_valida_argumentos(self):
        with self.assertRaises(ValueError):
            nms_por_clase(cajas([0, 0, 1, 1]), torch.tensor([0.5, 0.4]), torch.tensor([1]))
        with self.assertRaises(ValueError):
            nms_por_clase(cajas([0, 0, 1, 1]), torch.tensor([0.5]), torch.tensor([1]), iou_thresh=1.5)
        with self.assertRaises(ValueError):
            nms_por_clase(cajas([0, 0, 1, 1]), torch.tensor([0.5]), torch.tensor([1]), max_det_per_class=0)

    @unittest.skipUnless(torch.cuda.is_available(), "sin GPU")
    def test_gpu_igual_que_cpu(self):
        g = torch.Generator().manual_seed(0)
        xy = torch.rand(50, 2, generator=g) * 200
        b = torch.cat([xy, xy + 10 + torch.rand(50, 2, generator=g) * 40], dim=1)
        s, l = torch.rand(50, generator=g), torch.randint(1, 4, (50,), generator=g)
        cpu = nms_por_clase(b, s, l, max_det_per_class=None)
        gpu = nms_por_clase(b.cuda(), s.cuda(), l.cuda(), max_det_per_class=None)
        self.assertEqual(gpu.device.type, "cuda")
        self.assertEqual(cpu.tolist(), gpu.cpu().tolist())


@unittest.skipUnless(HAS_TORCHVISION, "torchvision no instalado")
class NMSContraTorchvision(unittest.TestCase):
    """torchvision se usa SOLO aquí, como referencia para verificar; el pipeline usa nms_por_clase."""

    def test_coincide_con_batched_nms(self):
        g = torch.Generator().manual_seed(42)
        for iou_thresh in (0.3, 0.5, 0.7):
            with self.subTest(iou_thresh=iou_thresh):
                xy = torch.rand(300, 2, generator=g) * 200
                wh = 5 + torch.rand(300, 2, generator=g) * 50
                b = torch.cat([xy, xy + wh], dim=1)
                s = torch.rand(300, generator=g)
                l = torch.randint(1, 4, (300,), generator=g)
                nuestro = nms_por_clase(b, s, l, iou_thresh=iou_thresh, max_det_per_class=None)
                ref = torchvision.ops.batched_nms(b, s, l, iou_thresh)
                self.assertEqual(nuestro.tolist(), ref.tolist())


def logit(p):
    return math.log(p / (1 - p))


def logits_con_duplicados():
    """Un mapa [1,8,16,16] de fondo con 3 celdas encendidas:
    dos celdas vecinas que describen casi la misma caja de sacro (duplicado) y una de coxal izquierdo."""
    out = torch.zeros(1, 8, 16, 16)
    out[:, 0] = -10.0                                                  # objectness ~ 0 en todo el mapa
    out[:, 3:5] = logit(0.25)                                          # cajas de 64 x 64 px
    out[:, 5:8] = torch.tensor([-5.0, -5.0, -5.0]).view(1, 3, 1, 1)
    # sacro (clase 0 -> label 1) en las celdas (8,5) y (8,6); los centros quedan a 3 px
    for (r, c, obj, tx) in ((8, 5, 5.0, 0.9), (8, 6, 4.0, 0.1)):
        out[0, 0, r, c] = obj
        out[0, 1, r, c] = logit(tx)
        out[0, 2, r, c] = 0.0                                          # ty = 0.5
        out[0, 5, r, c] = 5.0
    # coxal izquierdo (clase 1 -> label 2) en (3,12)
    out[0, 0, 3, 12] = 3.0
    out[0, 6, 3, 12] = 5.0
    return out


class NMSIntegradoConLaCabeza(unittest.TestCase):
    def test_postprocess_elimina_el_duplicado(self):
        out = logits_con_duplicados()
        raw = decode_predictions(out, score_thresh=0.05)[0]
        final = postprocess(out, score_thresh=0.05, nms_fn=nms_por_clase, **NMS_CONFIG)[0]
        self.assertEqual(raw["boxes"].shape[0], 3)                     # antes: 2 sacros + 1 coxal
        self.assertEqual(final["boxes"].shape[0], 2)                   # después: 1 por clase
        self.assertEqual(sorted(final["labels"].tolist()), [1, 2])
        self.assertEqual(final["class_probs"].shape, (2, 3))
        self.assertEqual(final["cells"][final["labels"] == 1].tolist(), [[8, 5]])   # gana la de mayor score
        for k, v in final.items():
            self.assertEqual(v.shape[0], 2, msg=f"la clave {k} no quedó alineada con las cajas")

    def test_mapa_sin_detecciones_da_salidas_vacias_con_forma(self):
        out = torch.full((2, 8, 16, 16), -10.0)
        for p in postprocess(out, score_thresh=0.05, nms_fn=nms_por_clase, **NMS_CONFIG):
            self.assertEqual(p["boxes"].shape, (0, 4))
            self.assertEqual(p["scores"].shape, (0,))
            self.assertEqual(p["labels"].shape, (0,))
            self.assertEqual(p["class_probs"].shape, (0, 3))


def predicted_boxes_top1_miguel(decoded_i, score_min=0.05):
    """Copia del marcador provisional de entrenamiento_multitarea (2).ipynb (celda 10), para compararlo."""
    boxes, labels = [], []
    for c in (1, 2, 3):
        idx = (decoded_i["labels"] == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        j = idx[decoded_i["scores"][idx].argmax()]
        if float(decoded_i["scores"][j]) >= score_min:
            boxes.append(decoded_i["boxes"][j]); labels.append(c)
    return (torch.stack(boxes) if boxes else torch.zeros(0, 4)), torch.tensor(labels, dtype=torch.long)


class NMSReemplazaAlMarcadorDeMiguel(unittest.TestCase):
    """Con max_det_per_class=1, nms_por_clase da las mismas cajas que el marcador usado al entrenar:
    se puede reemplazar sin reentrenar."""

    def test_mismas_cajas_en_salidas_aleatorias(self):
        g = torch.Generator().manual_seed(7)
        for _ in range(20):
            out = torch.randn(4, 8, 16, 16, generator=g) * 3
            raw = decode_predictions(out, score_thresh=0.0)
            fin = postprocess(out, score_thresh=0.05, nms_fn=nms_por_clase, iou_thresh=0.5, max_det_per_class=1)
            for r, f in zip(raw, fin):
                boxes_m, labels_m = predicted_boxes_top1_miguel(r)
                orden = torch.argsort(f["labels"])
                self.assertEqual(f["labels"][orden].tolist(), labels_m.tolist())
                torch.testing.assert_close(f["boxes"][orden], boxes_m)


if __name__ == "__main__":
    unittest.main()
