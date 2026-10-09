"""Pruebas de la unión detección -> NMS -> segmentación -> pertenencia (pipeline_pengwin.py). Autor: Fabián.

Usa piezas FALSAS con salidas controladas (un detector y un decoder que siempre "ven" lo mismo) para probar
la integración y las salidas vacías sin depender del entrenamiento. Las cabezas reales son la grid y la de
pertenencia; el backbone, el decoder, la restauración y el enlace 3D se simulan con su mismo contrato.

Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_pipeline -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distancias_fragmentos import measure_fragment_separations  # noqa: E402
from pertenencia import PertenenciaHead, clase_de_id  # noqa: E402
from pipeline_pengwin import (COLUMNAS_FRAGMENTOS, evaluar_pertenencia, inferir_cortes,  # noqa: E402
                              inferir_volumen, resumen_evaluacion_pertenencia)


# ------------------------------------------------------------------ piezas simuladas (mismo contrato)
class BackboneFalso(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 256, kernel_size=16, stride=16)


def features_falsas(backbone, x):
    B = x.shape[0]
    return [x.new_zeros(B, 32, 128, 128), x.new_zeros(B, 64, 64, 64), x.new_zeros(B, 128, 32, 32), backbone.conv(x)]


def logit(p):
    return math.log(p / (1 - p))


def logits_sacro_duplicado_y_coxal():
    """Sacro detectado por dos celdas vecinas (duplicado) + coxal izquierdo. Igual que en test_nms."""
    out = torch.zeros(1, 8, 16, 16)
    out[:, 0] = -10.0
    out[:, 3:5] = logit(0.25)
    out[:, 5:8] = -5.0
    for r, c, obj, tx in ((8, 5, 5.0, 0.9), (8, 6, 4.0, 0.1)):
        out[0, 0, r, c], out[0, 1, r, c], out[0, 5, r, c] = obj, logit(tx), 5.0
    out[0, 0, 3, 12], out[0, 6, 3, 12] = 3.0, 5.0
    return out


class DetectorFalso(nn.Module):
    def __init__(self, logits=None):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.logits = logits

    def forward(self, f4):
        B = f4.shape[0]
        if self.logits is None:
            return torch.full((B, 8, 16, 16), -10.0, device=f4.device)   # nada supera el umbral
        return self.logits.to(f4.device).expand(B, -1, -1, -1).clone()


def priors_falsos(detections, output_size=(256, 256), device=None):
    H, W = output_size
    pri = torch.zeros((len(detections), 3, H, W), device=device)
    for b, d in enumerate(detections):
        for box, lab in zip(d["boxes"].tolist(), d["labels"].tolist()):
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            pri[b, int(lab) - 1, max(y0, 0):y1, max(x0, 0):x1] = 1.0
    return pri


class DecoderFalso(nn.Module):
    """Guarda los priors que recibe y pasa la imagen; el 'decode' lee de la imagen qué corte es (píxel [0,0])."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.ultimos_priors = None

    def forward(self, image, features, priors, output_size=(256, 256)):
        self.ultimos_priors = priors.detach().cpu()
        return {"image": image}


def hacer_decode(mascaras_por_z):
    """mascaras_por_z: {z: máscara 2D con IDs del dataset}. El z se lee del píxel [0,0] de la imagen."""
    def decode(seg_out, center_threshold=0.25, min_center_distance=7, min_instance_pixels=1, batch_index=0):
        z = int(round(float(seg_out["image"][batch_index, 0, 0, 0])))
        inst = mascaras_por_z.get(z, np.zeros((256, 256), dtype=np.int64)).astype(np.uint16)
        sem = np.vectorize(clase_de_id)(inst.astype(np.int64)).astype(np.uint8)
        clases = {int(i): clase_de_id(int(i)) for i in np.unique(inst) if i != 0}
        return inst, sem, clases
    return decode


def restore_falso(label_mask, original_shape_yx, transform, output_size=256):
    return np.asarray(label_mask).astype(np.int32)                     # sin letterbox: identidad


def link_falso(inst_zyx, sem_zyx, min_overlap=0.10):
    """El mismo ID local en cortes distintos es el mismo fragmento 3D (suficiente para probar el contrato)."""
    ids = [int(v) for v in np.unique(inst_zyx) if v != 0]
    clases = {}
    for i in ids:
        vals, cnt = np.unique(sem_zyx[inst_zyx == i], return_counts=True)
        clases[i] = int(vals[np.argmax(cnt)])
    mapas = [{int(i): int(i) for i in np.unique(s) if i != 0} for s in inst_zyx]
    return inst_zyx.astype(np.uint32), clases, mapas, []


def preprocess_falso(corte):
    return np.asarray(corte, dtype=np.float32), {"scale": 1.0, "offset_x": 0, "offset_y": 0}


def componentes(det_logits=None, mascaras_por_z=None):
    torch.manual_seed(0)
    return {"backbone": BackboneFalso(), "cbam": None, "det_head": DetectorFalso(det_logits),
            "seg_decoder": DecoderFalso(), "cls_head": PertenenciaHead(),
            "features_fn": features_falsas, "priors_fn": priors_falsos,
            "decode_fn": hacer_decode(mascaras_por_z or {}), "restore_fn": restore_falso,
            "link_fn": link_falso, "preprocess_fn": preprocess_falso}


def imagenes(*zs):
    x = torch.zeros(len(zs), 1, 256, 256)
    for b, z in enumerate(zs):
        x[b, 0, 0, 0] = z
    return x


def fragmentos_en_cajas():
    m = np.zeros((256, 256), dtype=np.int64)
    m[120:150, 80:110] = 1                                             # dentro de la caja del sacro
    m[40:70, 180:220] = 12                                             # dentro de la caja del coxal izquierdo
    m[200:230, 10:40] = 25                                             # fuera de toda caja
    return m


# ------------------------------------------------------------------ pruebas por corte
class PorCorte(unittest.TestCase):
    def test_sin_detecciones_ni_instancias(self):
        r = inferir_cortes(componentes(), imagenes(0))[0]
        self.assertEqual(r["raw"]["boxes"].shape, (0, 4))
        self.assertEqual(r["final"]["boxes"].shape, (0, 4))
        self.assertEqual(r["final"]["class_probs"].shape, (0, 3))
        self.assertEqual(r["instancias"], [])
        self.assertEqual(r["instance_mask"].shape, (256, 256))
        self.assertEqual(int(r["class_map"].sum()), 0)

    def test_instancia_sin_caja_se_conserva(self):
        m = np.zeros((256, 256), dtype=np.int64)
        m[100:120, 100:120] = 3
        r = inferir_cortes(componentes(mascaras_por_z={0: m}), imagenes(0))[0]
        self.assertEqual(len(r["instancias"]), 1)                      # no se borra el fragmento
        self.assertTrue(r["instancias"][0]["sin_caja"])
        self.assertIn(r["instancias"][0]["clase"], (1, 2, 3))

    def test_union_completa(self):
        comp = componentes(logits_sacro_duplicado_y_coxal(), {0: fragmentos_en_cajas()})
        r = inferir_cortes(comp, imagenes(0))[0]
        self.assertEqual(r["raw"]["boxes"].shape[0], 3)                # antes del NMS: duplicado incluido
        self.assertEqual(r["final"]["boxes"].shape[0], 2)              # después: una caja por región
        # el decoder recibió priors hechos con las cajas FINALES: sacro y coxal izq. sí, coxal der. no
        pri = comp["seg_decoder"].ultimos_priors[0]
        self.assertGreater(pri[0].sum().item(), 0)
        self.assertGreater(pri[1].sum().item(), 0)
        self.assertEqual(pri[2].sum().item(), 0)
        por_id = {d["id"]: d for d in r["instancias"]}
        self.assertEqual(sorted(por_id), [1, 12, 25])
        self.assertEqual(por_id[1]["clase_caja"], 1)
        self.assertEqual(por_id[12]["clase_caja"], 2)
        self.assertTrue(por_id[25]["sin_caja"])
        for d in r["instancias"]:
            self.assertAlmostEqual(float(d["probs"].sum()), 1.0, places=5)
            self.assertTrue((r["class_map"][r["instance_mask"] == d["id"]] == d["clase"]).all())
            self.assertEqual(d["conflicto_semantica"], d["clase_semantica"] != d["clase"])

    def test_lote_mezclado(self):
        comp = componentes(mascaras_por_z={1: fragmentos_en_cajas()})
        res = inferir_cortes(comp, imagenes(0, 1, 0))
        self.assertEqual([len(r["instancias"]) for r in res], [0, 3, 0])

    def test_lote_vacio(self):
        self.assertEqual(inferir_cortes(componentes(), torch.zeros(0, 1, 256, 256)), [])

    def test_falta_un_componente(self):
        comp = componentes()
        comp["cls_head"] = None
        with self.assertRaises(KeyError):
            inferir_cortes(comp, imagenes(0))


# ------------------------------------------------------------------ pruebas por volumen
class PorVolumen(unittest.TestCase):
    def volumen(self, n=3):
        v = np.zeros((n, 256, 256), dtype=np.float32)
        for z in range(n):
            v[z, 0, 0] = z                                             # marca para que el decoder falso sepa el corte
        return v

    def test_volumen_vacio(self):
        out = inferir_volumen(componentes(), self.volumen(), (0.8, 0.8, 1.5), batch_size=2, case_id="000")
        self.assertEqual(out["instance_volume"].shape, (3, 256, 256))
        self.assertEqual(int(out["instance_volume"].max()), 0)
        self.assertEqual(list(out["fragmentos"].columns), COLUMNAS_FRAGMENTOS)
        self.assertEqual(len(out["fragmentos"]), 0)
        self.assertEqual(out["region_by_id"], {})
        self.assertEqual(len(out["cortes"]), 3)
        tabla = measure_fragment_separations(out["instance_volume"], out["spacing_xyz"],
                                             region_by_id=out["region_by_id"])
        self.assertEqual(len(tabla), 0)                                # Daniela recibe una tabla vacía, no un error

    def test_volumen_con_fragmentos_llega_a_distancias(self):
        a, b = np.zeros((256, 256), np.int64), np.zeros((256, 256), np.int64)
        a[100:140, 100:140] = 1
        a[150:190, 30:80] = 12
        b[100:140, 100:140] = 1
        b[100:110, 150:160] = 3                                        # fragmento pequeño solo en el corte 1
        comp = componentes(mascaras_por_z={0: a, 1: b, 2: a})
        out = inferir_volumen(comp, self.volumen(), (0.8, 0.8, 1.5), batch_size=2, case_id="001")
        frag = out["fragmentos"].set_index("fragment_id")
        self.assertEqual(sorted(frag.index), [1, 3, 12])
        self.assertEqual(int(frag.loc[1, "n_cortes"]), 3)
        self.assertEqual(int(frag.loc[3, "n_cortes"]), 1)
        self.assertEqual(int(frag.loc[1, "n_voxels"]), 3 * 40 * 40)
        probs = frag[["prob_sacro", "prob_coxal_izquierdo", "prob_coxal_derecho"]].to_numpy()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
        ids_volumen = {int(v) for v in np.unique(out["instance_volume"]) if v != 0}
        self.assertEqual(set(out["region_by_id"]), ids_volumen)         # cada fragmento tiene pertenencia
        tabla = measure_fragment_separations(out["instance_volume"], out["spacing_xyz"], case_id="001",
                                             region_by_id=out["region_by_id"])
        self.assertEqual(len(tabla), 3)
        self.assertTrue((tabla["distance_mm"] >= 0).all())


# ------------------------------------------------------------------ evaluación
class Evaluacion(unittest.TestCase):
    def setUp(self):
        self.comp = componentes(logits_sacro_duplicado_y_coxal(), {0: fragmentos_en_cajas()})
        self.targets = [{"case_id": "001", "slice_index": 10, "instance_mask": fragmentos_en_cajas()}]

    def test_modo_pred_y_resumen(self):
        df = evaluar_pertenencia(self.comp, imagenes(0), self.targets, modo="pred")
        self.assertEqual(len(df), 3)                                   # una fila por instancia REAL
        self.assertTrue((df["iou"] == 1.0).all())
        resumen = resumen_evaluacion_pertenencia(df).set_index("metodo")
        self.assertEqual(resumen.loc["regla_mayoria_semantica", "f1_macro"], 1.0)
        self.assertEqual(int(resumen.loc["regla_caja_detectada", "n_sin_clase"]), 1)   # el ID 25 no cae en caja

    def test_modo_gt(self):
        df = evaluar_pertenencia(self.comp, imagenes(0), self.targets, modo="gt")
        self.assertEqual(sorted(df["clase_real"].tolist()), [1, 2, 3])
        self.assertFalse(df["prob_sacro"].isna().any())

    def test_instancia_no_detectada_cuenta_como_error(self):
        comp = componentes(logits_sacro_duplicado_y_coxal(), {})        # el decoder no encuentra nada
        df = evaluar_pertenencia(comp, imagenes(0), self.targets, modo="pred")
        self.assertTrue((df["pred_cabeza"] == 0).all())
        self.assertIsInstance(resumen_evaluacion_pertenencia(df), pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
