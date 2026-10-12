"""Carga clases, funciones y constantes de los notebooks del proyecto sin ejecutar sus celdas.

Cada definicion compartida vive en un solo notebook de `notebooks/` (por ejemplo,
`PengwinDetectionDataset` en 06_targets_deteccion.ipynb). Los notebooks posteriores la cargan por
nombre, igual que `evaluacion.py` y los notebooks de entrenamiento, para no copiar codigo ni repetir
calculos pesados de los notebooks anteriores.
"""
import ast
import json
from pathlib import Path


def _target_names(node):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def _segmento(source, node):
    """Como ast.get_source_segment, pero sin perder los decoradores de funciones y clases.

    ast.get_source_segment empieza en la linea del `def`, asi que `@torch.no_grad()` se perdia y la funcion
    cargada calculaba gradientes (mas memoria y mas lenta en evaluacion). Solo se usa con nodos de nivel superior.
    """
    decoradores = getattr(node, "decorator_list", None)
    if not decoradores:
        return ast.get_source_segment(source, node)
    lineas = source.splitlines(keepends=True)
    inicio = min(d.lineno for d in decoradores) - 1
    return "".join(lineas[inicio:node.end_lineno]).rstrip("\n")


def cargar_definiciones(notebooks, definiciones, valores=(), destino=None):
    """Ejecuta en `destino` (el `globals()` del notebook) las definiciones pedidas.

    notebooks: ruta o lista de rutas .ipynb; si un nombre aparece en varios, gana el primero.
    definiciones: nombres de funciones o clases. valores: constantes asignadas a nivel superior.
    Tambien ejecuta los `import` de las celdas de donde sale cada definicion, para que las funciones
    encuentren sus dependencias (por ejemplo `maximum_filter`). No ejecuta celdas de prueba.
    Devuelve el numero de definiciones cargadas.
    """
    if destino is None:
        raise ValueError("Pasa destino=globals() para que las definiciones queden en el notebook")
    if isinstance(notebooks, (str, Path)):
        notebooks = [notebooks]
    wanted_defs, wanted_values = set(definiciones), set(valores)
    imports, snippets, found_defs, found_values = [], [], set(), set()

    for notebook_path in map(Path, notebooks):
        if not notebook_path.is_file():
            raise FileNotFoundError(f"No encuentro el notebook: {notebook_path}")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            cell_snippets = []
            for node in tree.body:
                name = getattr(node, "name", None)
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and name in wanted_defs and name not in found_defs):
                    cell_snippets.append(_segmento(source, node))
                    found_defs.add(name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    selected = _target_names(node) & wanted_values
                    if selected and not selected.issubset(found_values):
                        cell_snippets.append(ast.get_source_segment(source, node))
                        found_values.update(selected)
            if cell_snippets:
                imports += [ast.get_source_segment(source, node) for node in tree.body
                            if isinstance(node, (ast.Import, ast.ImportFrom))]
                snippets += cell_snippets

    missing_defs, missing_values = wanted_defs - found_defs, wanted_values - found_values
    if missing_defs or missing_values:
        nombres = ", ".join(Path(p).name for p in notebooks)
        raise RuntimeError(f"Faltan definiciones en {nombres}: "
                           f"funciones/clases={sorted(missing_defs)}, constantes={sorted(missing_values)}")
    exec("\n".join(dict.fromkeys(imports)) + "\n\n" + "\n\n".join(snippets), destino)
    return len(snippets)
