"""Medición volumétrica de separación entre fragmentos pélvicos.

Las etiquetas MHA del dataset identifican fragmentos con IDs 1..30. Para
predicciones con IDs propios, se debe proporcionar `region_by_id`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import ndimage


REGION_BY_DATASET_ID = {
    **{label_id: "sacro" for label_id in range(1, 11)},
    **{label_id: "coxal_izquierdo" for label_id in range(11, 21)},
    **{label_id: "coxal_derecho" for label_id in range(21, 31)},
}

DISTANCE_COLUMNS = [
    "case_id",
    "region",
    "fragment_id",
    "main_fragment_id",
    "is_main",
    "n_voxels",
    "distance_mm",
]


def measure_fragment_separations(
    instance_volume: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    *,
    case_id: str | None = None,
    region_by_id: Mapping[int, str] | None = None,
    main_id_by_region: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Mide la distancia mínima borde a borde entre máscaras voxelizadas.

    Args:
        instance_volume: Array entero (z, y, x); 0 es fondo y cada valor no
            nulo identifica un fragmento.
        spacing_xyz_mm: Espaciado de SimpleITK `GetSpacing()` en orden (x,y,z).
        case_id: Identificador opcional del caso para la tabla de salida.
        region_by_id: Mapeo ID de fragmento -> región anatómica. Si se omite,
            usa la codificación de etiquetas del dataset PENGWIN (1..30).
        main_id_by_region: Mapeo opcional región -> ID del fragmento principal.
            Si se omite, se elige el fragmento de mayor número de vóxeles en
            cada región, como en el EDA existente.

    Returns:
        DataFrame con una fila por fragmento. Se interpreta cada vóxel como
        una celda física de dimensiones dadas por el spacing. Dilatar la
        máscara principal una celda en cada eje y aplicar EDT calcula la
        separación euclidiana entre las celdas; máscaras en contacto dan 0 mm.
        El fragmento principal tiene distancia cero.
    """
    volume = np.asarray(instance_volume)
    if volume.ndim != 3:
        raise ValueError(f"Se esperaba volumen 3D (z, y, x); llegó shape={volume.shape}.")
    if not np.issubdtype(volume.dtype, np.integer):
        raise TypeError("instance_volume debe contener IDs enteros de instancia.")

    spacing_xyz = np.asarray(spacing_xyz_mm, dtype=float)
    if spacing_xyz.shape != (3,) or not np.isfinite(spacing_xyz).all() or (spacing_xyz <= 0).any():
        raise ValueError("spacing_xyz_mm debe tener tres valores positivos finitos en orden (x, y, z).")
    spacing_zyx = tuple(spacing_xyz[::-1])

    labels, counts = np.unique(volume[volume != 0], return_counts=True)
    if labels.size == 0:
        return pd.DataFrame(columns=DISTANCE_COLUMNS)

    if region_by_id is None:
        region_by_id = REGION_BY_DATASET_ID
    missing = [int(label) for label in labels if int(label) not in region_by_id]
    if missing:
        raise ValueError(
            "Falta la pertenencia anatómica para IDs de instancia: "
            f"{missing}. Proporciona region_by_id para las máscaras predichas."
        )

    fragments_by_region: dict[str, list[tuple[int, int]]] = {}
    for label, count in zip(labels, counts):
        region = region_by_id[int(label)]
        fragments_by_region.setdefault(region, []).append((int(label), int(count)))

    if main_id_by_region is None:
        selected_main = {
            region: min(fragments, key=lambda item: (-item[1], item[0]))[0]
            for region, fragments in fragments_by_region.items()
        }
    else:
        selected_main = dict(main_id_by_region)

    rows = []
    for region, fragments in fragments_by_region.items():
        if region not in selected_main:
            raise ValueError(f"No se indicó fragmento principal para la región {region!r}.")
        main_id = int(selected_main[region])
        if main_id not in {label for label, _ in fragments}:
            raise ValueError(f"El fragmento principal {main_id} no está presente en {region!r}.")

        region_ids = [label for label, _ in fragments]
        region_coordinates = np.where(np.isin(volume, region_ids))
        lower = [max(int(axis.min()) - 1, 0) for axis in region_coordinates]
        upper = [min(int(axis.max()) + 2, size) for axis, size in zip(region_coordinates, volume.shape)]
        crop_slices = tuple(slice(start, stop) for start, stop in zip(lower, upper))
        region_volume = volume[crop_slices]
        main_mask = region_volume == main_id
        # La dilatación Chebyshev de una celda representa los bordes físicos
        # de los vóxeles principales; EDT conserva el spacing anisotrópico.
        main_cells_with_border = ndimage.binary_dilation(
            main_mask, structure=np.ones((3, 3, 3), dtype=bool)
        )
        distance_map = ndimage.distance_transform_edt(
            ~main_cells_with_border, sampling=spacing_zyx
        )

        for fragment_id, n_voxels in fragments:
            is_main = fragment_id == main_id
            if is_main:
                distance_mm = 0.0
            else:
                fragment_mask = region_volume == fragment_id
                fragment_surface = fragment_mask & ~ndimage.binary_erosion(
                    fragment_mask,
                    structure=ndimage.generate_binary_structure(3, 1),
                    border_value=0,
                )
                distance_mm = float(distance_map[fragment_surface].min())

            rows.append({
                "case_id": case_id,
                "region": region,
                "fragment_id": fragment_id,
                "main_fragment_id": main_id,
                "is_main": is_main,
                "n_voxels": n_voxels,
                "distance_mm": distance_mm,
            })

    return pd.DataFrame(rows, columns=DISTANCE_COLUMNS).sort_values(
        ["region", "is_main", "fragment_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
