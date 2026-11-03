from __future__ import annotations

import json
from pathlib import Path


def extract_markov_boundary(
    edges: list[dict[str, object]],
    target: str,
    *,
    known_at_decision: set[str] | None = None,
) -> dict[str, object]:
    """Extract a temporal Markov-boundary hypothesis and leakage-safe state set.

    Lagged parents are safe state candidates. Contemporaneous parents are only
    included in the state set when explicitly declared known before the action.
    Children and spouses are reported for analysis but excluded from state by
    default because their measurement order can be ambiguous.
    """
    known_at_decision = known_at_decision or set()
    parents: set[tuple[str, int]] = set()
    children: set[str] = set()
    spouses: set[tuple[str, int]] = set()

    for edge in edges:
        source, child, lag = str(edge["source"]), str(edge["target"]), int(edge["lag"])
        if child == target:
            parents.add((source, lag))
        if source == target and lag == 0:
            children.add(child)

    for child in children:
        for edge in edges:
            if str(edge["target"]) != child:
                continue
            source, lag = str(edge["source"]), int(edge["lag"])
            if source != target:
                spouses.add((source, lag))

    boundary = parents | spouses | {(child, 0) for child in children}
    state = {
        node for node in parents
        if node[1] > 0 or (node[1] == 0 and node[0] in known_at_decision)
    }

    def records(nodes: set[tuple[str, int]]) -> list[dict[str, object]]:
        return [
            {"feature": feature, "lag": lag}
            for feature, lag in sorted(nodes, key=lambda value: (value[1], value[0]))
        ]

    return {
        "target": target,
        "parents": records(parents),
        "children": sorted(children),
        "spouses": records(spouses),
        "markov_boundary": records(boundary),
        "state_candidates": records(state),
        "state_rule": "lagged parents plus explicitly pre-decision contemporaneous parents",
    }


def save_feature_sets(
    dag_file: str | Path,
    output: str | Path,
    targets: list[str],
    known_at_decision: set[str] | None = None,
) -> dict[str, object]:
    dag = json.loads(Path(dag_file).read_text())
    if dag.get("converged") is not True or dag.get("instantaneous_is_dag") is not True:
        raise ValueError("NTS graph must converge and be acyclic before extracting features")
    result = {
        "source_dag": str(dag_file),
        "warning": "Markov boundaries inherit uncertainty from the learned observational graph.",
        "targets": {
            target: extract_markov_boundary(
                dag["edges"], target, known_at_decision=known_at_decision
            )
            for target in targets
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
