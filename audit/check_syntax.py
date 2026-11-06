from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parents[1]
rows = []
for folder in ["p4_design4"]:
    for p in sorted((root / folder).rglob("*.py")):
        source = p.read_text(encoding="utf-8-sig")
        ast.parse(source, filename=str(p))
        compile(source, str(p), "exec")
        rows.append(
            {
                "file": str(p),
                "syntax": "PASS",
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
ast.parse((root / "audit/mine_defaults.py").read_text(encoding="utf-8-sig"))
report = {
    "scope": "static syntax only; no imports, model execution, training, or dataset debugging",
    "python": rows,
    "mine_defaults_source": "PASS",
    "runtime_tests": "Static syntax check only; runtime and accuracy validation are separate.",
}
print(json.dumps(report, ensure_ascii=False, indent=2))
