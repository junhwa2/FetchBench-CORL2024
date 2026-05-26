import numpy as np
import pandas as pd

#import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Convert boolean types (True/False or np.bool_) to integer
# ---------------------------------------------------------
def convert_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    return x


# ---------------------------------------------------------
# Recursively flatten a nested dictionary
# Keys become: parent:child:subchild ...
# ---------------------------------------------------------
def flatten_dict(d, parent_key='', sep=':'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        # Case 1: nested dictionary
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))

        # Case 2: list/tuple/array
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.array(v, dtype=object)

            # Scalar-like array/list
            if arr.size == 1 and not isinstance(arr[0], dict):
                items[new_key] = convert_bool(arr.reshape(-1)[0])

            # Single dictionary inside list → flatten it
            elif arr.size == 1 and isinstance(arr[0], dict):
                items.update(flatten_dict(arr[0], new_key, sep=sep))

            # Multi-element array/list → convert booleans and store as comma-separated string
            else:
                converted = [convert_bool(x) for x in arr.tolist()]
                items[new_key] = ",".join(map(str, converted))

        # Case 3: direct scalar value
        else:
            items[new_key] = convert_bool(v)

    return items


# ---------------------------------------------------------
# Helpers for the new CSV layout
# ---------------------------------------------------------
def _flatten_value(key, val):
    """Wrap a single (key, val) into a flat dict using flatten_dict."""
    return flatten_dict({key: val})


def _eval_success_bool(val):
    """`data['success'][i]` may be a scalar, list, or ndarray (per env)."""
    arr = np.asarray(val).flatten()
    if arr.size == 0:
        return None
    return bool(arr.astype(bool).any())


# ---------------------------------------------------------
# Main function: Convert npy file to CSV
# Layout:
#   id, success (combined plan ∧ execution), label,
#   plan_success, plan_failure (only when not ok),
#   step{k}_success, step{k}_failure, step{k}_<other metrics>, ...
# - Columns that end up entirely empty/None across rows are dropped.
# - `*_status='ok'` semantics are converted to `*_success=1` (0 otherwise);
#   the original non-ok reason is preserved under `*_failure`.
# - The top-level `extra` dict is inlined (no `extra:` prefix).
# ---------------------------------------------------------
_STEP_KEEP_NAMES = (
    "plan_success", "plan_failure",
    "execute_success", "execute_failure",
)


def _is_kept_step_col(key):
    """Step{k}_* keys we keep in the CSV: plan/execute success/failure only.

    Exact-name match on what follows `step{k}_` so we don't accidentally keep
    debug metrics like `step0_grasp_plan_success` or `step0_fetch_plan_failure`.
    """
    if not key.startswith("step"):
        return False
    prefix = key.split("_", 1)[0]
    if not (prefix.startswith("step") and prefix[4:].isdigit()):
        return False
    rest = key[len(prefix) + 1:]  # strip "step{k}_"
    return rest in _STEP_KEEP_NAMES


def npy_to_csv(npy_path, csv_path):
    data = np.load(npy_path, allow_pickle=True).item()
    num_rows = len(next(iter(data.values())))

    rows = []
    for i in range(num_rows):
        row = {"id": i}

        extra = data["extra"][i] if "extra" in data and i < len(data["extra"]) else {}

        # Top-level: combined success comes from base eval (overridden by our
        # planner-aware eval to mean `motion_plan_success AND planner_held`).
        row["success"] = (int(_eval_success_bool(data["success"][i]))
                          if "success" in data else None)

        # bc_plan_completed = # of plan steps whose execute_success=1.
        # progress          = bc_plan_completed / bc_plan_length (float in [0,1]).
        # Both left as None when plan length is missing/zero.
        bc_len = extra.get("bc_plan_length")
        if isinstance(bc_len, (int, np.integer)) and bc_len > 0:
            n_ok = sum(
                1 for k, v in extra.items()
                if k.startswith("step")
                and k.split("_", 1)[0][4:].isdigit()
                and k[len(k.split("_", 1)[0]) + 1:] == "execute_success"
                and bool(convert_bool(v))
            )
            row["progress"]          = float(n_ok) / float(int(bc_len))
            row["bc_plan_completed"] = int(n_ok)
        else:
            row["progress"]          = None
            row["bc_plan_completed"] = None

        # Planner pipeline flags — written by solve(); always present once
        # the planner-aware branch has run.
        for k in ("bc_plan_length", "bc_plan_time",
                  "target_exist", "bc_plan_exist", "motion_plan_success"):
            if k in extra:
                row[k] = extra[k]

        # The compact plan trace, if any.
        if isinstance(extra.get("plan"), list):
            row["plan"] = " -> ".join(
                f"{int(o)}:[{','.join(str(int(g)) for g in gs)}]"
                for o, gs in extra["plan"]
            )

        # Per-step: only plan_success/plan_failure/execute_success/execute_failure.
        for k, v in extra.items():
            if _is_kept_step_col(k):
                row[k] = convert_bool(v) if not isinstance(v, str) else v

        rows.append(row)

    df = pd.DataFrame(rows)

    # Drop columns that are entirely empty across rows.
    def _all_empty(series):
        as_str = series.astype("object").apply(
            lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
        )
        return as_str.replace({"None": "", "nan": ""}).eq("").all()

    df = df.loc[:, [c for c in df.columns if not _all_empty(df[c])]]

    # Reorder: id, success, bc_plan_length, bc_plan_time, then planner flags,
    # then step{k}_* grouped (numeric step order, within-step in suffix order),
    # then plan trace + anything else.
    leading = [c for c in [
        "id", "success", "progress",
        "bc_plan_completed", "bc_plan_length", "bc_plan_time",
        "target_exist", "bc_plan_exist", "motion_plan_success",
    ] if c in df.columns]

    def _step_sort_key(c):
        prefix = c.split("_", 1)[0]
        step_n = int(prefix[4:]) if prefix[4:].isdigit() else 1_000_000
        rest = c[len(prefix) + 1:]
        suf_order = (_STEP_KEEP_NAMES.index(rest)
                     if rest in _STEP_KEEP_NAMES
                     else len(_STEP_KEEP_NAMES))
        return (step_n, suf_order, c)

    step_cols = sorted(
        [c for c in df.columns if c.startswith("step") and _is_kept_step_col(c)],
        key=_step_sort_key,
    )
    rest = [c for c in df.columns if c not in leading and c not in step_cols]
    df = df[leading + step_cols + rest]

    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    run_dir = './runs/RigidObjCellShelfDesk_0_FetchMeshCurobo_Debug_2025-12-16_14-43-25'
    npy_path = f'{run_dir}/result.npy'
    csv_path = f'{run_dir}/result.csv'
    npy_to_csv(npy_path, csv_path)