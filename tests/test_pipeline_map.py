"""Check docs/dev/pipeline_map.toml against the code.

The map lists every loader, matrix method, windowed engine and public
function, so that a change can find each parallel path it must cover. These
tests fail when the code and the map no longer agree; the message names each
entry to add, remove or correct in the map.
"""

import importlib
import inspect
import pathlib
import re
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "docs" / "dev" / "pipeline_map.toml"
if not MAP_PATH.exists():
    # The source distribution ships tests/ but not docs/, so the map exists
    # only in a repository checkout.
    pytest.skip(f"{MAP_PATH} is not in this tree", allow_module_level=True)
MAP = tomllib.loads(MAP_PATH.read_text())


def _module(name):
    # pg_gpu/__init__.py exports the function windowed_analysis under the
    # module's own name, so attribute access on the package can give the
    # function; import_module always gives the module.
    return importlib.import_module(f"pg_gpu.{name}")


def _resolve(name):
    """Resolve "module.attr[.attr]" or "Class.attr" to the object."""
    head, *rest = name.split(".")
    if head in MAP["matrices"]:
        obj = getattr(_module(MAP["matrices"][head]["module"]), head)
    else:
        obj = _module(head)
    for part in rest:
        obj = getattr(obj, part)
    return obj


def _diff(label, in_code, in_map):
    """A message naming the entries that only the code or only the map has."""
    lines = [f"{label}: the map and the code disagree. Edit "
             "docs/dev/pipeline_map.toml; docs/dev/README.md says how."]
    if in_code - in_map:
        lines.append(f"  add to the map: {sorted(in_code - in_map)}")
    if in_map - in_code:
        lines.append(f"  remove from the map: {sorted(in_map - in_code)}")
    return "\n".join(lines)


def _public_functions(module_name):
    mod = _module(module_name)
    return {f"{module_name}.{n}" for n, f in vars(mod).items()
            if not n.startswith("_") and inspect.isfunction(f)
            and f.__module__ == mod.__name__}


def _engine_stats(engine):
    """The statistics an engine computes, from the set its code uses."""
    stats = _resolve(MAP["engines"][engine]["stats"])
    return set(stats() if callable(stats) else stats)


# Engines with a set of their own; the streaming engine runs the others.
ENGINES = sorted(e for e, row in MAP["engines"].items() if "stats" in row)


def _strings(node):
    """Every string in a parsed TOML table, at any depth."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def test_every_module_is_mapped():
    in_code = {p.stem for p in (ROOT / "pg_gpu").glob("*.py")}
    meta = MAP["meta"]
    in_map = set(meta["inventory_modules"]) | set(meta["other_modules"])
    assert in_code == in_map, _diff("pg_gpu modules ([meta])", in_code, in_map)


def test_every_public_function_has_a_row():
    in_code = set().union(*map(_public_functions, MAP["meta"]["inventory_modules"]))
    in_map = set(MAP["functions"])
    assert in_code == in_map, _diff("public functions ([functions])", in_code, in_map)


@pytest.mark.parametrize("name", sorted(MAP["functions"]))
def test_axis_kwargs_match_signature(name):
    params = inspect.signature(_resolve(name)).parameters
    in_code = {p for p in MAP["meta"]["axis_kwargs"] if p in params}
    in_map = set(MAP["functions"][name]["axis_kwargs"])
    assert in_code == in_map, _diff(f"axis_kwargs of {name}", in_code, in_map)


def test_every_loader_has_a_row():
    in_code = set()
    for cls_name, row in MAP["matrices"].items():
        cls = getattr(_module(row["module"]), cls_name)
        in_code |= {f"{cls_name}.{n}" for n in dir(cls) if n.startswith("from_")}
    in_map = {n for n in MAP["inputs"] if not n.endswith(".__init__")}
    assert in_code == in_map, _diff("loaders ([inputs])", in_code, in_map)


@pytest.mark.parametrize("name", sorted(MAP["inputs"]))
def test_loader_params_match_signature(name):
    params = list(inspect.signature(_resolve(name)).parameters)
    params = [p for p in params if p not in ("self", "cls")]
    assert params == MAP["inputs"][name]["params"], (
        f"params of {name} in [inputs]: the code has {params}")


@pytest.mark.parametrize("cls_name", sorted(MAP["matrices"]))
def test_matrix_methods(cls_name):
    row = MAP["matrices"][cls_name]
    cls = getattr(_module(row["module"]), cls_name)
    in_code = {n for n in dir(cls) if not n.startswith("_")}
    in_map = set(row["methods"])
    assert in_code == in_map, _diff(f"methods of {cls_name} ([matrices])", in_code, in_map)


def test_windowed_names():
    in_code = set().union(*map(_engine_stats, ENGINES))
    in_map = set(MAP["windowed"])
    assert in_code == in_map, _diff("windowed statistics ([windowed])", in_code, in_map)


@pytest.mark.parametrize("engine", ENGINES)
def test_windowed_engines(engine):
    in_code = _engine_stats(engine)
    in_map = {s for s, row in MAP["windowed"].items() if engine in row["engines"]}
    assert in_code == in_map, _diff(
        f"statistics that list engine {engine!r} in [windowed]", in_code, in_map)


def test_windowed_rows_name_known_engines():
    named = {e for row in MAP["windowed"].values() for e in row["engines"]}
    unknown = named - set(ENGINES)
    assert not unknown, f"[windowed] rows name engines with no `stats` in [engines]: {sorted(unknown)}"


def test_alias_rows_match():
    # An alias is the same function object under a second name, so its row
    # must say the same as the row of the name it aliases; only notes differ.
    groups = {}
    for name in MAP["functions"]:
        groups.setdefault(id(_resolve(name)), []).append(name)
    for names in (g for g in groups.values() if len(g) > 1):
        first, *rest = sorted(names)
        expect = {k: v for k, v in MAP["functions"][first].items() if k != "notes"}
        for name in rest:
            got = {k: v for k, v in MAP["functions"][name].items() if k != "notes"}
            differ = sorted(k for k in expect.keys() | got.keys() if expect.get(k) != got.get(k))
            assert not differ, f"{name} aliases {first}, but their rows differ in {differ}"


def test_every_named_path_resolves():
    # A dotted name that starts with a module or a matrix class is a code
    # path, in any field of the map, prose included. ".py" ends a file name.
    heads = (set(MAP["meta"]["inventory_modules"]) | set(MAP["meta"]["other_modules"])
             | set(MAP["matrices"]))
    pattern = re.compile(r"(?<![\w.])((?:%s)(?:\.\w+)+)" % "|".join(sorted(heads)))
    names = {m for text in _strings(MAP) for m in pattern.findall(text)
             if not m.endswith(".py")}
    broken = []
    for name in sorted(names):
        try:
            _resolve(name)
        except (AttributeError, ModuleNotFoundError):
            broken.append(name)
    assert not broken, "names in the map that the code does not have:\n  " + "\n  ".join(broken)
