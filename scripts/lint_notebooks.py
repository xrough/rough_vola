#!/usr/bin/env python3
"""
lint_notebooks.py — enforce authoring rules for the archived recipe notebooks.

The rules exist so the series works as a template: every recipe must be
findable, self-contained, and paste-able into another project. Prose promises
that; this script checks it.

    python3 scripts/lint_notebooks.py                 # structural checks
    python3 scripts/lint_notebooks.py --copy-paste-test   # + execute each recipe alone
    python3 scripts/lint_notebooks.py --quiet         # errors only

Active research notebooks do not use recipe cards and are outside this check.
Exit code is non-zero if any rule is violated.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks/legacy"

MAX_CODE_LINES = 25
# A notebook now carries two or three former ones, so the budget is per notebook,
# not per topic: three 60s sections is a legitimate 180s, and anything past that is
# a recipe doing too much work to be copied out.
MAX_RUNTIME_S = 200.0
COPY_PASTE_TIMEOUT_S = 15 * 60
RECIPE_HEADING = re.compile(r"^##\s+(R\d+\.\d+)\s+·\s+(.+?)\s*$", re.M)
ANCHOR = re.compile(r'<a\s+id="([a-z0-9\-]+)"\s*>\s*</a>')
REQUIRED_SECTIONS = ("**Use when**", "**Inputs**", "**Needs**", "**Gotchas**",
                     "**See also**")
NEEDS = re.compile(r"\*\*Needs\*\*\s*(.+?)\s*$", re.M)
STRUCTURAL_TAGS = {"setup", "parameters", "swap-data", "section-data"}
# A structural cell is a manifest, not logic: the parameters cell of a notebook covering
# three sections carries three sections' knobs, and a section's data cell carries its
# loads. Holding those to the 25-line limit would push configuration into rvlab, which is
# the opposite of the point — so they get their own, larger cap.
MAX_STRUCTURAL_LINES = 45
DATA_TAGS = {"swap-data", "section-data"}

# The standalone validator deliberately uses smaller synthetic fixtures than
# the presentation notebooks. It is checking name/dependency portability, not
# reproducing benchmark-quality estimates hundreds of times. Values are only
# capped when the corresponding parameter exists; no new notebook API is
# invented, and small user-authored values remain unchanged.
COPY_PASTE_LIMITS = {
    "N_DATES": 180,
    "POSITION_N_DATES": 180,
    "N_ENTITIES": 120,
    "PANEL_DATES": 180,
    "PANEL_ENTITIES": 40,
    "N_PATHS": 120,
    "N_STEPS": 128,
    "N_BOOT": 60,
    "N_SAMPLE": 2_500,
    "N_SAMPLES": 80,
    "CLASSIFICATION_SAMPLES": 1_500,
    "N_TRIALS": 3,
    "MAX_TIMESTAMPS": 600,
    "N_REPEATS": 2,
    "FRAGMENT_IDS": 120,
    "N_TRAIN": 160,
    "N_TEST": 60,
}


def source(cell) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def recipe_paths() -> list[Path]:
    """The archival order defines the series; additional studies stay separate."""
    order = NOTEBOOKS / ".order"
    if not order.exists():
        return sorted(NOTEBOOKS.glob("*.ipynb"))
    names = dict.fromkeys(line.strip() for line in order.read_text().splitlines()
                         if line.strip() and not line.lstrip().startswith("#"))
    return [NOTEBOOKS / name for name in names if (NOTEBOOKS / name).exists()]


def tags(cell) -> list[str]:
    return list(cell.get("metadata", {}).get("tags", []))


def anchor_for(recipe_id: str) -> str:
    """R6.4 -> r6-4"""
    return recipe_id.lower().replace(".", "-")


def ends_in_bare_assignment(code: str) -> bool:
    """True when the final statement produces no visible output."""
    body = [ln for ln in code.rstrip().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not body:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    if not tree.body:
        return False
    last = tree.body[-1]
    if isinstance(last, (ast.Expr, ast.Import, ast.ImportFrom, ast.FunctionDef,
                         ast.ClassDef, ast.For, ast.While, ast.If, ast.With,
                         ast.Try, ast.Assert, ast.Raise, ast.Pass)):
        return False
    return isinstance(last, (ast.Assign, ast.AugAssign, ast.AnnAssign))


def lint_notebook(path: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one notebook."""
    errors: list[str] = []
    warnings: list[str] = []
    nb = json.loads(path.read_text())
    cells = nb.get("cells", [])
    name = path.name

    md = [c for c in cells if c.get("cell_type") == "markdown"]
    code = [c for c in cells if c.get("cell_type") == "code"]
    all_md = "\n".join(source(c) for c in md)

    # ── required tagged cells ────────────────────────────────────────────────
    for tag, label in (("setup", "setup"), ("parameters", "parameters"),
                       ("swap-data", "swap-data")):
        n = sum(1 for c in code if tag in tags(c))
        if n != 1:
            errors.append(f"{name}: expected exactly 1 cell tagged '{label}', found {n}")

    # A notebook is a small document with a stable interface, not merely a bag
    # of valid recipe cards.  These headings are what make every file navigable
    # in the same way and what the README promises readers they will find.
    for heading in ("# ", "## The question", "## Parameters",
                    "## Where the data comes from", "## Recipes in this notebook",
                    "## Things to watch", "## Verdict & what carries forward"):
        if heading not in all_md:
            errors.append(f"{name}: missing skeleton heading {heading!r}")

    # ── recipe cards ─────────────────────────────────────────────────────────
    recipes = RECIPE_HEADING.findall(all_md)
    if not recipes:
        errors.append(f"{name}: no '## R<n>.<m> · title' recipe headings found")

    seen_ids = set()
    for cell in md:
        text = source(cell)
        found = RECIPE_HEADING.findall(text)
        if not found:
            continue
        rid, title = found[0]
        if rid in seen_ids:
            errors.append(f"{name}: duplicate recipe id {rid}")
        seen_ids.add(rid)

        anchors = ANCHOR.findall(text)
        want = anchor_for(rid)
        if want not in anchors:
            errors.append(f"{name}: {rid} ('{title}') missing anchor <a id=\"{want}\"></a>")
        for section in REQUIRED_SECTIONS:
            if section not in text:
                errors.append(f"{name}: {rid} missing {section} in its recipe card")

    # ── cell-level rules ─────────────────────────────────────────────────────
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = source(cell)
        lines = [ln for ln in src.splitlines() if ln.strip()]
        structural = bool(STRUCTURAL_TAGS & set(tags(cell)))
        cap = MAX_STRUCTURAL_LINES if structural else MAX_CODE_LINES
        if len(lines) > cap:
            errors.append(f"{name}: code cell {i} has {len(lines)} lines (max {cap}); "
                          + ("split the section" if structural
                             else "move the logic into rvlab"))
        if cell.get("execution_count") is None:
            errors.append(f"{name}: code cell {i} has no stored execution; run top to bottom")
        if any(out.get("output_type") == "error" for out in cell.get("outputs", [])):
            errors.append(f"{name}: code cell {i} contains a stored exception")
        # Structural cells are exempt from the "end in something visible" rule:
        # a papermill `parameters` cell is assignments by definition, and setup /
        # swap-data exist to bind names, not to display results.
        if STRUCTURAL_TAGS & set(tags(cell)):
            continue
        if ends_in_bare_assignment(src):
            errors.append(f"{name}: code cell {i} ends in a bare assignment; "
                          "end with something visible (a head(), figure or metric)")

    # ── narrative before code ────────────────────────────────────────────────
    kinds = [c.get("cell_type") for c in cells]
    for i in range(1, len(kinds)):
        if kinds[i] == "code" and kinds[i - 1] == "code":
            a, b = cells[i - 1], cells[i]
            if not (tags(a) or tags(b)):        # setup/parameters/swap-data may pair up
                warnings.append(f"{name}: code cells {i-1} and {i} are adjacent "
                                "with no markdown between them")

    # ── runtime budget ───────────────────────────────────────────────────────
    total = 0.0
    for cell in code:
        meta = cell.get("metadata", {}).get("execution", {})
        start, end = meta.get("iopub.execute_input"), meta.get("shell.execute_reply")
        if start and end:
            from datetime import datetime
            total += (datetime.fromisoformat(end.replace("Z", "+00:00"))
                      - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
    if total > MAX_RUNTIME_S:
        warnings.append(f"{name}: recorded runtime {total:.1f}s exceeds "
                        f"the {MAX_RUNTIME_S:.0f}s budget")

    return errors, warnings


def _anchor_map() -> dict[str, set[str]]:
    """notebook filename -> the set of <a id="..."> anchors it defines."""
    available: dict[str, set[str]] = {}
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        text = "\n".join(source(c) for c in json.loads(nb_path.read_text())["cells"]
                         if c.get("cell_type") == "markdown")
        available[nb_path.name] = set(ANCHOR.findall(text))
    return available


LINK = re.compile(r"\]\(([\w\.\-]+\.ipynb)(?:#([a-z0-9\-]+))?\)")
RECIPE_LINK = re.compile(
    r"\[(R\d+\.\d+)\]\((?:(\w[\w.\-]*\.ipynb))?#(r\d+\-\d+)\)")
# A link with no file part points inside the document it is written in: a "See also"
# to another recipe in the same notebook, or the README's own contents line.
SELF_LINK = re.compile(r"\]\(#([a-z0-9\-]+)\)")
HEADING = re.compile(r"^#{2,3}\s+(.+?)\s*$", re.M)


def _heading_slug(text: str) -> str:
    """GitHub's heading-anchor rule, for links into README headings."""
    return re.sub(r"[^a-z0-9 -]", "", text.lower()).replace(" ", "-")


def check_links() -> list[str]:
    """Every notebook link, in the README *and* between notebooks, must resolve.

    Cross-notebook links are how the series is navigated — a "See also" pointing at
    a renamed file is worse than no link, because it looks authoritative. Checking
    them here is what makes renaming a notebook a safe operation.
    """
    errors = []
    available = _anchor_map()

    sources: list[tuple[str, str]] = []
    readme = NOTEBOOKS / "README.md"
    if not readme.exists():
        return [f"{NOTEBOOKS / 'README.md'} is missing"]
    sources.append(("README.md", readme.read_text()))
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        text = "\n".join(source(c) for c in json.loads(nb_path.read_text())["cells"]
                         if c.get("cell_type") == "markdown")
        sources.append((nb_path.name, text))

    for origin, text in sources:
        for target, anchor in LINK.findall(text):
            if target not in available:
                errors.append(f"{origin}: links to unknown notebook {target}")
            elif anchor and anchor not in available[target]:
                errors.append(f"{origin}: links to {target}#{anchor}, which does not exist")

        # Same-document links. In a notebook the targets are <a id="..."> anchors;
        # in the README they are its own headings, which is what the index's
        # contents line points at.
        own = (available[origin] if origin in available
               else {_heading_slug(h) for h in HEADING.findall(text)})
        for anchor in SELF_LINK.findall(text):
            if anchor not in own:
                errors.append(f"{origin}: links to #{anchor}, which it does not define")

        # A link can resolve and still lie about where it goes: `[R2.4](#r3-4)`
        # looks authoritative and is especially hard to notice in a long index.
        for label, target, anchor in RECIPE_LINK.findall(text):
            labelled_anchor = label.lower().replace(".", "-")
            if labelled_anchor != anchor:
                destination = f"{target}#{anchor}" if target else f"#{anchor}"
                errors.append(
                    f"{origin}: link label {label} points to {destination}; "
                    f"expected anchor #{labelled_anchor}")
    return errors


def check_collection() -> list[str]:
    """Check the reading order and recipe namespace across the whole series."""
    errors: list[str] = []
    order_path = NOTEBOOKS / ".order"
    actual = {p.name for p in recipe_paths()}
    if not order_path.exists():
        return [f"{order_path} is missing"]

    listed = [line.strip() for line in order_path.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    duplicates = sorted({name for name in listed if listed.count(name) > 1})
    if duplicates:
        errors.append(f"{order_path}: duplicate notebooks: {duplicates}")
    missing = sorted(actual - set(listed))
    unknown = sorted(set(listed) - actual)
    if missing:
        errors.append(f"{order_path}: omitted notebooks: {missing}")
    if unknown:
        errors.append(f"{order_path}: missing notebooks: {unknown}")

    owners: dict[str, str] = {}
    for position, name in enumerate(listed, start=1):
        path = NOTEBOOKS / name
        if not path.exists():
            continue
        nb = json.loads(path.read_text())
        all_md = "\n".join(source(c) for c in nb.get("cells", [])
                           if c.get("cell_type") == "markdown")
        recipe_ids = [rid for rid, _ in RECIPE_HEADING.findall(all_md)]
        prefixes = {int(rid[1:].split(".")[0]) for rid in recipe_ids}
        if recipe_ids and prefixes != {position}:
            errors.append(
                f"{name}: recipes use prefixes {sorted(prefixes)}, "
                f"but .order position is {position}")
        for rid in recipe_ids:
            if rid in owners:
                errors.append(f"duplicate recipe id {rid}: {owners[rid]} and {name}")
            owners[rid] = name
    return errors


def copy_paste_test(path: Path, quiet: bool) -> list[str]:
    """Execute each *recipe* alone, with only its own section's context in scope.

    The recipe — its card plus the code cells under it — is the unit of reuse, so
    that is the unit tested. A recipe may tell its story in two code cells with an
    explanation between them; what it may not do is depend on a name defined by a
    *different* recipe. That is the property that makes these cells portable, and
    it is exactly what this check enforces.

    Context is setup + parameters + the **nearest preceding data cell**. A notebook
    covering several sections rebinds its frames at each section head, exactly as
    the reader experiences it. Declared prerequisites first run with *their* data
    context; the target section's data context is restored before its own code.
    """
    import io
    from contextlib import redirect_stderr, redirect_stdout

    nb = json.loads(path.read_text())
    base: list[str] = []                       # setup + parameters, always in scope
    order: list[str] = []
    blocks: dict[str, list[str]] = {}
    needs: dict[str, list[str]] = {}
    section_of: dict[str, str | None] = {}     # recipe -> its section's data cell
    current = None
    current_data: str | None = None

    for cell in nb["cells"]:
        if cell.get("cell_type") == "markdown":
            text = source(cell)
            found = RECIPE_HEADING.findall(text)
            if found:
                current = found[0][0]
                order.append(current)
                blocks[current] = []
                section_of[current] = current_data
                declared = NEEDS.search(text)
                needs[current] = re.findall(r"R\d+\.\d+", declared.group(1)) if declared else []
            continue
        if cell.get("cell_type") != "code":
            continue
        cell_tags = set(tags(cell))
        if DATA_TAGS & cell_tags:
            current_data = source(cell)        # later sections shadow earlier ones
        elif STRUCTURAL_TAGS & cell_tags:
            base.append(source(cell))
        elif current:
            blocks[current].append(source(cell))

    def chain(rid, seen=None):
        """Declared prerequisites, transitively, in notebook order."""
        seen = seen if seen is not None else set()
        for dep in needs.get(rid, []):
            if dep not in seen and dep in blocks:
                seen.add(dep)
                chain(dep, seen)
        return [r for r in order if r in seen]

    # `display` is an IPython builtin: present in a real notebook, absent from a
    # bare exec. Providing it keeps the check honest — a recipe that uses it does
    # work when pasted into a notebook, which is the claim being tested.
    try:
        from IPython.display import display as _display
    except ImportError:
        _display = print

    errors = []
    for rid in order:
        if not blocks[rid]:
            continue
        env: dict = {"__name__": "__main__", "display": _display}
        buf = io.StringIO()
        failed = None
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                for block in base:
                    exec(compile(block, f"<{path.name}:preamble>", "exec"), env)
                # Apply the validator profile *before* the section data cell so
                # a 400 x 500 presentation panel does not get rebuilt for every
                # standalone recipe. Parameters remain representative and above
                # every notebook's minimum window/fold sizes.
                for name, limit in COPY_PASTE_LIMITS.items():
                    value = env.get(name)
                    if isinstance(value, int) and not isinstance(value, bool):
                        env[name] = min(value, limit)
                env["N_JOBS"] = 1
                env["FAST_MODE"] = True
                env["SHOW_FULL_CATALOGUE"] = False
                active_data = None

                def load_context(owner: str) -> None:
                    """Switch to the data cell belonging to ``owner`` if needed."""
                    nonlocal active_data
                    data = section_of.get(owner)
                    if data and data != active_data:
                        exec(compile(data, f"<{path.name}:{owner}:data>", "exec"), env)
                        active_data = data
                    # A data cell should not alter execution-policy parameters,
                    # but make the invariant explicit for custom adaptations.
                    env["N_JOBS"] = 1

                for dep in chain(rid):
                    load_context(dep)
                    for block in blocks[dep]:
                        exec(compile(block, f"<{path.name}:{dep}>", "exec"), env)
                load_context(rid)
                for j, block in enumerate(blocks[rid]):
                    exec(compile(block, f"<{path.name}:{rid}#{j}>", "exec"), env)
        except Exception as exc:
            failed = f"{type(exc).__name__}: {exc}"
            errors.append(
                f"{path.name}: recipe {rid} fails with only its declared prerequisites "
                f"{chain(rid) or '(none)'} ({failed}) — declare the missing one in **Needs**")
        finally:
            # Recipe environments are intentionally disposable. Close pyplot's
            # process-global figure registry as well, otherwise a chart-heavy
            # notebook retains every figure until its worker exits.
            try:
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass
            env.clear()
            gc.collect()
        if not quiet:
            deps = f" (after {', '.join(chain(rid))})" if chain(rid) else ""
            print(f"    {rid:>7s} {'FAIL' if failed else 'ok'}{deps}")
    return errors


def copy_paste_subprocess(path: Path, quiet: bool, output_root: Path) -> list[str]:
    """Run one notebook's recipe checks in a constrained child process.

    Notebook recipes legitimately import native numerical libraries and some
    launch joblib/OpenMP workers. Keeping notebooks in separate processes makes
    a native crash attributable, while single-process/thread environment caps
    keep this authoring check reliable on CI and Kaggle-sized machines.
    """
    notebook_output = output_root / path.stem
    notebook_output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "RVLAB_FORCE_SYNTHETIC": "1",
        "RVLAB_OUTPUT_DIR": str(notebook_output),
        "JOBLIB_TEMP_FOLDER": str(notebook_output / "joblib"),
        "LOKY_MAX_CPU_COUNT": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONFAULTHANDLER": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(notebook_output / "matplotlib"),
        "IPYTHONDIR": str(notebook_output / "ipython"),
        "XDG_CACHE_HOME": str(notebook_output / "cache"),
    })
    command = [sys.executable, str(Path(__file__).resolve()),
               "--_copy-paste-worker", str(path.resolve())]
    if quiet:
        command.insert(2, "--quiet")
    configured_timeout = os.environ.get("RVLAB_COPY_PASTE_TIMEOUT_S", "")
    try:
        timeout_s = (float(configured_timeout) if configured_timeout
                     else COPY_PASTE_TIMEOUT_S)
        if timeout_s <= 0:
            raise ValueError
    except ValueError:
        return ["RVLAB_COPY_PASTE_TIMEOUT_S must be a positive number of seconds"]

    try:
        result = subprocess.run(
            command, env=env, cwd=REPO, capture_output=True, text=True,
            timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        parts = []
        for value in (exc.stdout, exc.stderr):
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            if value and value.strip():
                parts.append(value.strip())
        output = "\n".join(parts)
        detail = "\n      ".join(output.splitlines()[-20:]) if output else "no worker output"
        return [
            f"{path.name}: copy-paste worker timed out after {timeout_s:g}s\n"
            f"      {detail}"
        ]

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr)
                       if part.strip())
    if result.returncode == 0:
        if output and not quiet:
            print(output)
        return []

    # Negative return codes are signals on POSIX; shell-reported 128+signal
    # values are left intact on platforms that use those instead.
    cause = (f"signal {-result.returncode}" if result.returncode < 0
             else f"exit {result.returncode}")
    detail = "\n      ".join(output.splitlines()[-20:]) if output else "no worker output"
    return [f"{path.name}: copy-paste worker failed ({cause})\n      {detail}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy-paste-test", action="store_true",
                    help="execute each recipe cell in isolation (slow)")
    ap.add_argument("--legacy", action="store_true",
                    help="select the archived recipe series (the default)")
    ap.add_argument("--quiet", action="store_true", help="print errors only")
    ap.add_argument("--_copy-paste-worker", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("notebooks", nargs="*", help="specific recipe notebooks (default: notebooks/legacy)")
    args = ap.parse_args()

    paths = ([Path(p) for p in args.notebooks] if args.notebooks
             else recipe_paths())
    if not paths:
        print("no notebooks found", file=sys.stderr)
        return 1

    if args._copy_paste_worker:
        if len(paths) != 1:
            print("copy-paste worker requires exactly one notebook", file=sys.stderr)
            return 2
        errors = copy_paste_test(paths[0], args.quiet)
        for error in errors:
            print(f"  ERROR  : {error}", file=sys.stderr)
        return 1 if errors else 0

    all_errors, all_warnings = [], []
    with tempfile.TemporaryDirectory(prefix="rvlab-copy-paste-") as temp_output:
        output_root = Path(temp_output)
        for path in paths:
            errors, warnings = lint_notebook(path)
            if args.copy_paste_test:
                if not args.quiet:
                    print(f"  copy-paste test: {path.name}")
                errors += copy_paste_subprocess(path, args.quiet, output_root)
            all_errors += errors
            all_warnings += warnings
            if not args.quiet:
                status = "FAIL" if errors else "ok"
                print(f"{status:>5}  {path.name}  "
                      f"({len(errors)} errors, {len(warnings)} warnings)")

    all_errors += check_links()
    all_errors += check_collection()

    for w in all_warnings:
        print(f"  warning: {w}")
    for e in all_errors:
        print(f"  ERROR  : {e}", file=sys.stderr)

    print(f"\n{len(paths)} notebooks · {len(all_errors)} errors · {len(all_warnings)} warnings")
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
