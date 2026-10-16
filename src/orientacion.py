"""Orientacion canonica de los volumenes CT del proyecto PENGWIN.

Problema: 34 de los 100 casos vienen en orientacion RAS (ejes x e y volteados: rotacion de 180 grados en el
plano axial) y 66 en LPS (identidad). Si se cargan "tal cual", la red ve cortes rotados o espejados segun el caso.

Solucion: al cargar, permutar/voltear los ejes del array para que SIEMPRE se cumpla
    eje x del array -> izquierda del paciente (L)
    eje y del array -> posterior (P)
    eje z del array -> superior (S)
es decir, direccion = identidad (LPS) para todos los casos. Es solo reordenar el array: no interpola ni cambia
ningun valor. Es el caso de los 100 casos del dataset (todos alineados a los ejes).

Si llega un volumen con direccion OBLICUA (p. ej. inclinacion del gantry o una rotacion arbitraria), no basta con
reordenar ejes: se remuestrea a una rejilla alineada (SciPy; trilineal para HU, vecino mas cercano para etiquetas)
y se avisa con un warning. Ese camino interpola, asi que las etiquetas pueden cambiar en el borde de cada fragmento.

Convenciones (las de SimpleITK):
    array (z, y, x)  |  espaciado, origen y direccion en orden (x, y, z)  |  fisico = origen + D @ (espaciado * indice_xyz)

Uso tipico:
    from orientacion import load_canonical
    volumen, espaciado, origen = load_canonical(ruta_imagen, is_label=False)
    etiquetas, _, _            = load_canonical(ruta_etiqueta, is_label=True)

Probar el modulo (solo necesita NumPy):   python orientacion.py
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["axis_map", "is_axis_aligned", "is_canonical", "canonicalize_zyx", "resample_oblique_zyx",
           "restore_orientation_zyx", "load_canonical", "read_direction"]

_LETTERS = {(0, 1): "L", (0, -1): "R", (1, 1): "P", (1, -1): "A", (2, 1): "S", (2, -1): "I"}


def _as_matrix(direction):
    return np.asarray(direction, dtype=float).reshape(3, 3)


def axis_map(direction):
    """Letra fisica (LPS) hacia la que crece cada eje del array (x, y, z), y la desviacion respecto a ejes puros.
    Identidad -> ('LPS', 0.0). Los casos volteados de PENGWIN dan 'RAS'."""
    D = _as_matrix(direction)
    code, off_axis = "", 0.0
    for k in range(3):
        a = int(np.argmax(np.abs(D[:, k])))
        code += _LETTERS[(a, 1 if D[a, k] > 0 else -1)]
        off_axis = max(off_axis, 1.0 - abs(D[a, k]))
    return code, off_axis


def is_axis_aligned(direction, tol=1e-3):
    """True si cada eje del array coincide con un eje fisico (permutacion con signo, sin oblicuidad)."""
    D = _as_matrix(direction)
    return bool(np.all(1.0 - np.abs(D).max(0) < tol) and len(set(np.abs(D).argmax(0))) == 3)


def is_canonical(direction, atol=1e-4):
    return bool(np.allclose(_as_matrix(direction), np.eye(3), atol=atol))


def _axes_info(direction):
    D = _as_matrix(direction)
    if not is_axis_aligned(D):
        raise NotImplementedError(
            f"Direccion oblicua (desviacion {axis_map(D)[1]:.3f}): solo se puede revertir/reordenar sin remuestrear "
            "cuando los ejes estan alineados.")
    rows = np.abs(D).argmax(0)                         # eje fisico (0=L, 1=P, 2=S) de cada eje k del array
    signs = np.sign(D[rows, np.arange(3)])             # +1 si el eje crece hacia L/P/S, -1 si hacia R/A/I
    return D, rows, signs


def resample_oblique_zyx(arr_zyx, spacing_xyz, origin_xyz, direction, order=None, cval=None):
    """Remuestrea un volumen con direccion oblicua a una rejilla LPS (direccion = identidad) que lo contiene.
    order: 0 vecino mas cercano (por defecto para enteros/etiquetas) o 1 trilineal (por defecto para HU).
    cval: valor fuera del volumen original (0 en etiquetas; el minimo del volumen en imagenes).
    Devuelve (array (z,y,x), espaciado (x,y,z), origen (x,y,z)). El espaciado de salida de cada eje fisico es el del
    eje del array mas alineado con el."""
    from scipy import ndimage
    arr = np.asarray(arr_zyx)
    D = _as_matrix(direction)
    if not np.allclose(D @ D.T, np.eye(3), atol=1e-3):
        raise ValueError("La matriz de direccion no es ortonormal; el archivo parece corrupto")
    sp, org = np.asarray(spacing_xyz, dtype=float), np.asarray(origin_xyz, dtype=float)
    is_int = np.issubdtype(arr.dtype, np.integer)
    order = (0 if is_int else 1) if order is None else int(order)
    cval = (0 if is_int else float(arr.min())) if cval is None else cval

    n = np.array(arr.shape[::-1])                                        # (nx, ny, nz)
    corner_idx = np.array([[i, j, k] for i in (0, n[0] - 1) for j in (0, n[1] - 1) for k in (0, n[2] - 1)], dtype=float)
    corners = org + (corner_idx * sp) @ D.T                              # centros de vóxel de las 8 esquinas
    lo, hi = corners.min(0), corners.max(0)
    sp_out = sp[np.abs(D).argmax(1)]                                     # eje del array mas alineado con cada eje fisico
    size = np.floor((hi - lo) / sp_out + 1e-6).astype(int) + 1           # (nx', ny', nz')

    # indice_entrada_xyz = A @ indice_salida_xyz + t     (D es ortonormal: D^-1 = D^T)
    A = (D.T * sp_out[None, :]) / sp[:, None]
    t = (D.T @ (lo - org)) / sp
    out = ndimage.affine_transform(arr, A[::-1, ::-1], offset=t[::-1], output_shape=tuple(int(v) for v in size[::-1]),
                                   order=order, mode="constant", cval=cval, prefilter=False)
    return np.ascontiguousarray(out), tuple(float(v) for v in sp_out), tuple(float(v) for v in lo)


def canonicalize_zyx(arr_zyx, spacing_xyz, origin_xyz, direction, allow_resample=True, order=None, cval=None):
    """Lleva un volumen a orientacion canonica (LPS). Devuelve (array (z,y,x), espaciado (x,y,z), origen (x,y,z)).
    Direcciones alineadas a los ejes: solo se reordena (exacto). Oblicuas: se remuestrea (con warning) si
    allow_resample=True; si es False lanza NotImplementedError."""
    arr_zyx = np.asarray(arr_zyx)
    if arr_zyx.ndim != 3:
        raise ValueError("Se esperaba un volumen 3D con forma (z, y, x)")
    if not is_axis_aligned(direction):
        if not allow_resample:
            raise NotImplementedError(f"Direccion oblicua (desviacion {axis_map(direction)[1]:.3f}) y allow_resample=False")
        warnings.warn(f"Direccion oblicua (desviacion {axis_map(direction)[1]:.3f}): se remuestrea a rejilla LPS; "
                      "los bordes de las etiquetas pueden cambiar.", RuntimeWarning, stacklevel=2)
        return resample_oblique_zyx(arr_zyx, spacing_xyz, origin_xyz, direction, order=order, cval=cval)
    D, rows, signs = _axes_info(direction)
    sp, org = np.asarray(spacing_xyz, dtype=float), np.asarray(origin_xyz, dtype=float)
    k_of = [int(np.where(rows == p)[0][0]) for p in range(3)]     # eje del array que pasa a ser el fisico p
    a = np.transpose(arr_zyx, (2, 1, 0))                          # ejes (x, y, z) del array
    out = np.transpose(a, k_of)
    for p in range(3):
        if signs[k_of[p]] < 0:
            out = np.flip(out, axis=p)
    idx0 = np.array([a.shape[k] - 1 if signs[k] < 0 else 0 for k in range(3)], dtype=float)
    new_origin = org + D @ (sp * idx0)                            # posicion fisica del nuevo vóxel (0, 0, 0)
    return np.ascontiguousarray(np.transpose(out, (2, 1, 0))), tuple(float(v) for v in sp[k_of]), tuple(float(v) for v in new_origin)


def restore_orientation_zyx(arr_canonical_zyx, original_direction):
    """Inversa de canonicalize_zyx para un array: devuelve el array en la orientacion del archivo original.
    Util solo si hay que entregar resultados en la geometria original; el espaciado y el origen originales
    se leen del propio archivo. Para entrenar y evaluar no hace falta."""
    arr = np.asarray(arr_canonical_zyx)
    _, rows, signs = _axes_info(original_direction)
    out = np.transpose(np.transpose(arr, (2, 1, 0)), [int(r) for r in rows])   # eje k del original = eje fisico rows[k]
    for k in range(3):
        if signs[k] < 0:
            out = np.flip(out, axis=k)
    return np.ascontiguousarray(np.transpose(out, (2, 1, 0)))


def read_direction(path, resolver=None):
    """Direccion (9 valores) de un .mha sin leer los pixeles."""
    import SimpleITK as sitk
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(resolver(path) if resolver else path))
    reader.ReadImageInformation()
    return reader.GetDirection()


def load_canonical(path, is_label=False, resolver=None, return_direction=False, dtype=None):
    """Lee un .mha y lo devuelve en orientacion canonica: (array (z,y,x), espaciado (x,y,z), origen (x,y,z)).
    dtype=None -> int16 si is_label, si no float32 (HU). dtype="native" conserva el tipo del archivo (p. ej. int16 en HU:
    la mitad de memoria que float32; convertir solo el corte que se use). Cualquier otro dtype de NumPy tambien vale.
    `resolver` convierte rutas relativas del repositorio (p. ej. resolve_repo_path).
    return_direction=True añade al final la direccion ORIGINAL del archivo (para restore_orientation_zyx)."""
    import SimpleITK as sitk
    image = sitk.ReadImage(str(resolver(path) if resolver else path))
    arr = sitk.GetArrayFromImage(image)
    if not (isinstance(dtype, str) and dtype == "native"):
        arr = arr.astype(dtype if dtype is not None else (np.int16 if is_label else np.float32), copy=False)
    direction = image.GetDirection()
    if is_canonical(direction):
        result = (arr, tuple(float(v) for v in image.GetSpacing()), tuple(float(v) for v in image.GetOrigin()))
    else:
        result = canonicalize_zyx(arr, image.GetSpacing(), image.GetOrigin(), direction)
    return (*result, direction) if return_direction else result


# ---------------------------------------------------------------------------------------------------
def _rotation(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _selftest_oblique():
    """Volumen adquirido con direccion oblicua (inclinacion + giro): al canonizar debe coincidir con la verdad fisica."""
    rng = np.random.default_rng(1)
    D = _rotation(np.deg2rad(18), np.deg2rad(-7), np.deg2rad(25))
    sp, org, shape = np.array([0.8, 0.8, 1.2]), np.array([-30.0, 12.0, 5.0]), (50, 70, 80)       # (z, y, x)

    def physical(idx_xyz, o, s, M):
        return o + (idx_xyz * s) @ M.T

    zz, yy, xx = np.meshgrid(*[np.arange(m) for m in shape], indexing="ij")
    idx = np.stack([xx, yy, zz], -1).reshape(-1, 3).astype(float)
    P = physical(idx, org, sp, D)
    centre = P.mean(0)

    def label_truth(Q):                                    # dos "fragmentos" como cajas alineadas a los ejes fisicos
        a = np.all(np.abs(Q - (centre + [-12, 0, 0])) < [10, 9, 9], axis=-1)
        b = np.all(np.abs(Q - (centre + [14, 3, 4])) < [8, 8, 8], axis=-1)
        return np.where(a, 1, np.where(b, 11, 0)).astype(np.int16)

    def image_truth(Q):                                    # funcion lineal: la interpolacion trilineal es exacta
        return (0.5 * Q[..., 0] - 0.3 * Q[..., 1] + 0.8 * Q[..., 2]).astype(np.float32)

    lab_obl = label_truth(P).reshape(shape)
    img_obl = image_truth(P).reshape(shape)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lab_c, sp_c, org_c = canonicalize_zyx(lab_obl, sp, org, D.ravel())
    assert any(issubclass(x.category, RuntimeWarning) for x in w), "debio avisar del remuestreo"
    img_c, sp_i, org_i = canonicalize_zyx(img_obl, sp, org, D.ravel())
    assert np.allclose(sp_c, sp_i) and np.allclose(org_c, org_i)

    jz, jy, jx = np.meshgrid(*[np.arange(m) for m in lab_c.shape], indexing="ij")
    Pc = np.asarray(org_c) + np.stack([jx, jy, jz], -1) * np.asarray(sp_c)          # posicion fisica de la rejilla nueva
    # solo vóxeles bien dentro del volumen original (a >= 3 vóxeles de cualquier borde)
    idx_in = ((Pc - org) @ D) / sp
    inside = np.all((idx_in >= 3) & (idx_in <= np.array(shape[::-1]) - 4), axis=-1)
    exp_lab = label_truth(Pc)
    for cid in (1, 11):
        g, p = (exp_lab == cid) & inside, (lab_c == cid) & inside
        dice = 2 * (g & p).sum() / (g.sum() + p.sum())
        cg, cp = Pc[g].mean(0), Pc[p].mean(0)
        assert dice > 0.90 and np.linalg.norm(cg - cp) < 1.0, (cid, dice, np.linalg.norm(cg - cp))   # el vecino mas cercano mueve el borde hasta ~1 vóxel
    err = np.abs(img_c - image_truth(Pc))[inside]
    assert err.max() < 0.05, err.max()
    assert lab_c.dtype == np.int16 and img_c.dtype == np.float32
    print(f"OK: volumen oblicuo (18, -7, 25 grados) remuestreado: Dice de etiquetas > 0.90 y centroides a < 1 mm, error maximo de HU {err.max():.4f}.")


def _selftest():
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 30, size=(7, 9, 11)).astype(np.int16)           # (z, y, x), formas distintas a proposito
    sp_c, org_c = (0.8, 0.9, 1.0), (10.0, -5.0, 3.0)
    perms = [(0, 1, 2), (1, 0, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1)]
    n = 0
    for p in perms:
        for signs in np.ndindex(2, 2, 2):
            s = [1 if v == 0 else -1 for v in signs]
            D = np.zeros((3, 3))
            for k in range(3):
                D[p[k], k] = s[k]
            # array "original" con direccion D, construido a partir de una cuadricula canonica conocida
            a = np.transpose(np.transpose(vol, (2, 1, 0)), p)
            for k in range(3):
                if s[k] < 0:
                    a = np.flip(a, axis=k)
            arr = np.ascontiguousarray(np.transpose(a, (2, 1, 0)))
            sp = tuple(np.array(sp_c)[list(p)])
            assert is_axis_aligned(D) and axis_map(D)[1] < 1e-9
            can, sp2, org2 = canonicalize_zyx(arr, sp, org_c, D.ravel())
            assert is_canonical(np.eye(3).ravel())
            assert can.shape == vol.shape and np.allclose(sp2, sp_c), (p, s)
            # 1) mismo contenido que la cuadricula canonica, 2) misma posicion fisica de cada voxel
            assert np.array_equal(can, vol), (p, s)
            for _ in range(50):
                idx = np.array([rng.integers(m) for m in arr.shape[::-1]])          # (x, y, z) en el array original
                P = np.array(org_c) + D @ (np.array(sp) * idx)
                j = np.rint((P - np.array(org2)) / np.array(sp2)).astype(int)
                assert arr[idx[2], idx[1], idx[0]] == can[j[2], j[1], j[0]], (p, s)
            # 3) la inversa devuelve exactamente el array original
            assert np.array_equal(restore_orientation_zyx(can, D.ravel()), arr), (p, s)
            n += 1
    assert axis_map(np.eye(3).ravel()) == ("LPS", 0.0)
    assert axis_map(np.diag([-1.0, -1.0, 1.0]).ravel())[0] == "RAS"
    assert not is_axis_aligned(np.array([np.cos(0.35), -np.sin(0.35), 0, np.sin(0.35), np.cos(0.35), 0, 0, 0, 1]))
    try:
        canonicalize_zyx(vol, sp_c, org_c, _rotation(0.0, 0.0, 0.35).ravel(), allow_resample=False)
        raise AssertionError("debio rechazar la direccion oblicua con allow_resample=False")
    except (NotImplementedError, ValueError):
        pass
    _selftest_oblique()
    print(f"OK: {n} orientaciones (permutaciones x signos) canonizadas, con posicion fisica conservada e inversa exacta.")


if __name__ == "__main__":
    _selftest()
