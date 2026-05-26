"""Backward-chaining singulation planner.

Adapted from third_party/2025ICRA_PL/singulation/backward_chaining.py and
extended to consume the grasp KB schema produced by
`FetchMeshCuroboGORun.build_grasp_kb`:

    kb[obj_id] = {
        'status':        'graspable' | 'obstructed' | 'ungraspable',
        'clauses':       List[List[int]],   # OR of AND-clauses (ICRA format)
        'grasp_indices': List[List[int]],   # parallel to clauses
    }

Compared to the upstream implementation, this module:
  * Treats `status == 'ungraspable'` (and missing kb keys) as dead-ends — both
    when given as the goal and when reached as a child literal during search.
  * Uses `kb[o]['clauses']` instead of `kb[o]` directly; an empty clause list
    paired with `status == 'graspable'` still means "no LHS, done".
  * Drops the demo globals and unused `Node2` / `search` / `get_graspable_object`.
  * Adds `plan_with_grasps(kb, goal)` that post-processes the obj sequence
    into a list of `(obj_id, grasp_id)` pairs.
  * Adds `draw_and_or_graph(kb, goal, save_path)` to dump a Graphviz AND-OR
    diagram of the KB rooted at `goal`.
"""
import os
import subprocess
import time
from itertools import product


class Node:
    """Search node: literals still owed (`current`) + already committed (`parents`)."""

    def __init__(self, current, parents):
        self.current = list(current)
        self.parents = list(parents)
        self.cost = len(current) + len(parents)

    def __eq__(self, other):
        return self.current == other.current and self.parents == other.parents

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        return str(tuple(self.current))


def _clauses_of(kb, obj_id):
    """Return the list of AND-clauses for `obj_id`, or `None` if dead-end."""
    entry = kb.get(obj_id)
    if entry is None or entry.get('status') == 'ungraspable':
        return None
    return entry.get('clauses', [])


def conjugate_dnfs(dnfs):
    """Distribute-law product of multiple DNFs into one simplified DNF.

    Each input dnf is a list of AND-clauses (each clause = list of literals).
    The output is a list of AND-clauses representing the conjunction of all
    input DNFs, with duplicate literals removed inside each clause.
    """
    if not dnfs:
        return []
    tmp = []
    for combo in product(*[range(len(d)) for d in dnfs]):
        clause = []
        for j, d in enumerate(dnfs):
            clause.append(d[combo[j]])
        tmp.append(clause)
    out = []
    for d in tmp:
        merged = set()
        for sub in d:
            merged |= set(sub)
        out.append(list(merged))
    return out


def backward_chaining(kb, goal, time_limit=2.0, debug=False):
    """Best-first backward-chaining search.

    Args:
        kb: KB dict with the schema described at the top of this module.
        goal: list of obj_ids that must all become graspable.
        time_limit: seconds before the search aborts.
        debug: verbose prints.

    Returns:
        (ok, failed, sequence, expansions)
            ok=False, failed=True  → goal ungraspable or time-out
            ok=True,  failed=False → sequence is the execution order
                                     (leaves first, goal last)
            ok=True,  failed=True  → search exhausted without finding goal
    """
    # Guard: goal must be reachable in principle.
    for g in goal:
        if _clauses_of(kb, g) is None:
            return False, True, [], 0

    init_node = Node(set(goal), [])
    open_list = [init_node]
    closed_list = []
    cnt = 0
    start_t = time.time()

    while open_list:
        cnt += 1
        cur_node = min(open_list)
        open_list.remove(cur_node)
        closed_list.append(cur_node)

        if debug:
            print(f"[bc] cur={cur_node}  cost={cur_node.cost}")

        # Collect LHS of any literal in current node that still needs clauses.
        cur_node_lhs = []
        for lit in cur_node.current:
            clauses = _clauses_of(kb, lit)
            if clauses is None:
                # An ungraspable literal slipped in; this branch is doomed.
                cur_node_lhs = None
                break
            if clauses:                       # obstructed → has clauses to expand
                cur_node_lhs.append(clauses)
            # graspable → contributes no LHS; treated as already-satisfied
        if cur_node_lhs is None:
            continue
        if len(cur_node_lhs) == 0:
            return True, False, (cur_node.parents + cur_node.current)[::-1], cnt

        if time.time() - start_t > time_limit:
            return False, True, [], cnt

        # Expand: distribute-law product of all LHS DNFs → child clauses.
        for child in conjugate_dnfs(cur_node_lhs):
            child_parents = cur_node.parents + cur_node.current
            child_current = set(child)

            # Skip children that touch ungraspable / unknown literals.
            if any(_clauses_of(kb, lit) is None for lit in child_current):
                if debug:
                    print(f"[bc]   drop (ungraspable literal) {child_current}")
                continue

            # Loop check: literal already committed AND still obstructed.
            is_loop = False
            for lit in list(child_current):
                if lit in child_parents:
                    clauses = _clauses_of(kb, lit)
                    if clauses:               # obstructed → real loop
                        is_loop = True
                        break
                    else:                      # graspable now → drop from parents
                        child_parents.remove(lit)
            if is_loop:
                if debug:
                    print(f"[bc]   drop (loop) {child_current}")
                continue

            child_node = Node(child_current, child_parents)
            if child_node in closed_list or child_node in open_list:
                continue
            open_list.append(child_node)

    return True, True, [], cnt


def plan_with_grasps(kb, goal_id, time_limit=2.0, debug=False):
    """Run backward chaining and attach grasp ids to each obj in the sequence.

    For each step we expose grasps from EVERY clause whose obstacle set has
    already been cleared by the preceding steps — not just one clause. The
    KB's parallel structure (`clauses[i]` ↔ `grasp_indices[i]`) means a
    grasp is only physically feasible once its specific obstacle set is out
    of the way; including grasps from unsatisfied clauses would hand cuRobo
    grasps that still collide with leftover objs. Multiple satisfied
    clauses → their grasps are unioned, giving motion planning a wider
    feasible pool.

    Returns:
        (ok, plan)
            ok=False, plan=[]                          → no feasible plan
            ok=True,  plan=[(obj_id, [grasp_id, ...])] → execution order
    """
    ok, failed, sequence, _ = backward_chaining(
        kb, [goal_id], time_limit=time_limit, debug=debug)
    if not ok or failed:
        return False, []

    cleared = set()
    plan = []
    for o in sequence:
        entry = kb[o]
        if entry['status'] == 'graspable':
            # Empty-obstacle bucket (the only one for graspable status).
            grasp_ids = sorted({int(g) for g in entry['grasp_indices'][0]})
        else:
            # Union grasps from every clause whose obstacles ⊆ cleared.
            feasible = set()
            for clause, bucket in zip(entry['clauses'], entry['grasp_indices']):
                if {int(x) for x in clause}.issubset(cleared):
                    feasible.update(int(g) for g in bucket)
            if not feasible:
                # Sequence guarantees ≥1 satisfied clause; bail if not.
                return False, []
            grasp_ids = sorted(feasible)
        plan.append((int(o), grasp_ids))
        cleared.add(int(o))
    return True, plan


# ---------------------------------------------------------------------------
# Plan-step palette — shared color language across the AND-OR graph and the
# trimesh kb_vis render. The *target* (last step of the plan) is always red;
# earlier steps cycle through `_STEP_PALETTE` in order. Two renders should be
# read side-by-side: same step → same color in both.
# ---------------------------------------------------------------------------
# ColorBrewer "Set1" categorical palette — chosen for max hue separation
# (gold/magenta dropped: they collided with orange/red respectively; green
# and brown added as the most distinguishable replacements).
_TARGET_RGB = (228,  26,  28)         # red
_STEP_PALETTE = [
    ( 55, 126, 184),   # blue
    ( 77, 175,  74),   # green
    (255, 127,   0),   # orange
    (152,  78, 163),   # purple
    (  0, 170, 170),   # teal
    (166,  86,  40),   # brown
]


def step_color_rgb(step_idx, n_steps):
    """Return (r, g, b) for a step. Last step (== target) is always red."""
    if step_idx == n_steps - 1:
        return _TARGET_RGB
    return _STEP_PALETTE[step_idx % len(_STEP_PALETTE)]


def step_color_hex(step_idx, n_steps):
    r, g, b = step_color_rgb(step_idx, n_steps)
    return f"#{r:02x}{g:02x}{b:02x}"


def chosen_clauses(kb, plan):
    """Given the plan, return {obj_id: clause_idx_used} for obstructed obj.

    Mirrors the lookup in `plan_with_grasps`: at step k the chosen clause is
    the first one whose obstacle set is a subset of objects already cleared
    by steps 0..k-1. Graspable obj have no clause to pick (returns 0 for
    uniformity if you want to highlight grasp bucket).
    """
    cleared = set()
    chosen = {}
    for obj_id, _ in plan:
        entry = kb.get(obj_id)
        if entry is None:
            cleared.add(int(obj_id))
            continue
        if entry['status'] == 'graspable':
            chosen[int(obj_id)] = 0
        else:
            for idx, clause in enumerate(entry['clauses']):
                if set(int(x) for x in clause).issubset(cleared):
                    chosen[int(obj_id)] = idx
                    break
        cleared.add(int(obj_id))
    return chosen


# ---------------------------------------------------------------------------
# AND-OR graph visualization (Graphviz DOT). Each AND-clause is one node.
# ---------------------------------------------------------------------------

# Per-status *text* colors. Status is conveyed by font color only; nodes keep
# a uniform white fill with a black border so the graph stays monochrome except
# for the object labels.
_STATUS_TXT = {
    'graspable':   '#2e7d32',  # green
    'obstructed':  '#e59400',  # amber
    'ungraspable': '#c62828',  # red
}
_DEFAULT_TXT = '#37474f'


def _status_of(kb, o):
    entry = kb.get(o)
    return entry['status'] if entry else 'ungraspable'


def _obj_cell(o, status, step_color_hex=None):
    """HTML-label fragment: 'obj N' over '[status]' (two stacked lines).

    Two-line layout (`<br/>` between name and status) keeps each cell
    visually compact. Composite AND nodes wrap these cells in a single-row
    `<table>` so the row never wraps to a second line — see `_emit_dot`.
    The shared color is the step palette when this obj is in the plan
    (matches the kb_vis tint / grasp marker); otherwise the status color.
    """
    from html import escape
    col = step_color_hex if step_color_hex else _STATUS_TXT.get(status, _DEFAULT_TXT)
    return (f'<font color="{col}"><b>obj {escape(str(o))}</b><br/>'
            f'<font point-size="8">[{escape(status)}]</font></font>')


def _emit_dot(kb, goal, plan=None):
    """Build the DOT source for the AND-OR graph rooted at `goal`.

    Compressed layout — never draws the same obstructed obj twice:
      * Goal obj                    → standalone ellipse (root).
      * Single-member clause [X]    → direct edge parent → obj_X ellipse.
                                      If X is obstructed, its clauses then
                                      hang off obj_X.
      * Multi-member clause [X, Y]  → composite ellipse labeled
                                      `obj X ∧ obj Y` (one row, each cell 2
                                      lines: obj name / [status]).
      * Composite's obstructed members → NOT drawn as their own ellipse;
                                      their clauses attach directly under
                                      the composite. So `obj 13` inside a
                                      composite (5 ∧ 13) skips its own
                                      ellipse and exposes `obj 13`'s
                                      requirements (e.g. obj 5) hanging off
                                      the composite.

    Plan emphasis — peripheries=2 (double outline) on:
      * standalone ellipses for plan obj
      * the composite ellipse that corresponds to the plan's chosen clause
        for an obstructed plan obj
    The outline thickness itself stays uniform (penwidth=1.5).
    """
    # dpi=200 → ~2× sharper PNG (default 96). nodesep/ranksep keep enough
    # whitespace that one-line ellipses don't crowd each other horizontally.
    # All node/edge penwidths are unified at 1.5; plan emphasis uses
    # peripheries (double oval) instead.
    lines = ['digraph AndOr {',
             '  graph [rankdir=TB, fontname="Helvetica", charset="UTF-8",'
             ' bgcolor="white", dpi=200, nodesep=0.35, ranksep=0.5];',
             '  node  [fontname="Helvetica", fontsize=10, shape=ellipse,'
             ' style="filled", fillcolor="white", color="black", penwidth=1.5];',
             '  edge  [fontname="Helvetica", fontsize=9, color="black",'
             ' penwidth=1.5];']

    visited_obj    = set()
    visited_clause = set()
    visited_edge   = set()    # dedup parent→child edges

    plan = plan or []
    n_steps     = len(plan)
    step_of_obj = {int(o): i for i, (o, _) in enumerate(plan)}
    chosen      = chosen_clauses(kb, plan) if plan else {}
    BLACK       = '#000000'

    def step_col_for(x):
        if int(x) in step_of_obj:
            return step_color_hex(step_of_obj[int(x)], n_steps)
        return None

    def obj_node_id(o):
        return f"obj_{o}"

    def clause_node_id(parent_obj, idx):
        return f"cl_{parent_obj}_{idx}"

    def add_edge(src, dst, arrowhead_none=False):
        key = (src, dst, arrowhead_none)
        if key in visited_edge:
            return
        visited_edge.add(key)
        attrs = '[arrowhead=none]' if arrowhead_none else ''
        lines.append(f'  {src} -> {dst} {attrs};')

    def emit_obj_box(o, on_plan_path):
        """Render obj `o` as a standalone ellipse. Idempotent.

        on_plan_path=True → double oval (peripheries=2), step-color font.
        on_plan_path=False → single oval, all text in black.
        """
        if o in visited_obj:
            return
        visited_obj.add(o)
        status = _status_of(kb, o)
        if on_plan_path:
            col = step_col_for(o) or _STATUS_TXT.get(status, _DEFAULT_TXT)
        else:
            col = BLACK
        per = 2 if on_plan_path else 1
        lines.append(
            f'  {obj_node_id(o)} [peripheries={per}, '
            f'label=<{_obj_cell(o, status, step_color_hex=col)}>];'
        )

    def emit_composite(cid, clause, on_plan_path):
        """Composite ellipse ('obj X ∧ obj Y'). Idempotent.

        on_plan_path=True → double oval, each member cell in its step color
            (falls back to status color when the member isn't in the plan).
        on_plan_path=False → single oval, every cell in black.
        """
        if cid in visited_clause:
            return
        visited_clause.add(cid)
        cells = []
        for x in clause:
            x_status = _status_of(kb, x)
            if on_plan_path:
                col = step_col_for(x) or _STATUS_TXT.get(x_status, _DEFAULT_TXT)
            else:
                col = BLACK
            cells.append(_obj_cell(x, x_status, step_color_hex=col))
        wedge = '<td><font color="black"><b>&#8743;</b></font></td>'
        tds = wedge.join(f'<td>{c}</td>' for c in cells)
        inner = (f'<table border="0" cellspacing="2" cellborder="0">'
                 f'<tr>{tds}</tr></table>')
        per = 2 if on_plan_path else 1
        lines.append(
            f'  {cid} [shape=ellipse, peripheries={per}, label=<{inner}>];'
        )

    def expand_obstructed(parent_dot_id, obj_id, parent_on_plan_path):
        """Attach `obj_id`'s clearance (its clauses' sub-trees) below
        `parent_dot_id`. The plan-path flag is propagated only along the
        chosen clause of an in-plan obj — siblings (non-chosen OR clauses)
        and detours into off-plan obj branches reset to off-plan styling.
        """
        entry = kb.get(obj_id)
        if entry is None or _status_of(kb, obj_id) != 'obstructed':
            return
        chosen_idx = chosen.get(int(obj_id))
        obj_in_plan = int(obj_id) in step_of_obj

        for ci, clause in enumerate(entry['clauses']):
            if not clause:
                continue
            # A child clause stays on the plan path only when the parent was
            # on the plan path, this obj is itself in the plan, and we're
            # walking through the clause the plan actually picked.
            child_on_path = (parent_on_plan_path and obj_in_plan
                             and chosen_idx == ci)

            if len(clause) == 1:
                child = clause[0]
                emit_obj_box(child, on_plan_path=child_on_path)
                add_edge(parent_dot_id, obj_node_id(child))
                if _status_of(kb, child) == 'obstructed':
                    expand_obstructed(obj_node_id(child), child,
                                      parent_on_plan_path=child_on_path)
            else:
                cid = clause_node_id(obj_id, ci)
                emit_composite(cid, clause, on_plan_path=child_on_path)
                add_edge(parent_dot_id, cid, arrowhead_none=True)
                # Each obstructed member of this composite hangs its own
                # clearance directly off the composite. They inherit the
                # current child_on_path flag (off-plan composite → always
                # off-plan for its descendants).
                for sub_member in clause:
                    if _status_of(kb, sub_member) == 'obstructed':
                        expand_obstructed(cid, sub_member,
                                          parent_on_plan_path=child_on_path)

    for g in goal:
        g_on_path = int(g) in step_of_obj
        emit_obj_box(g, on_plan_path=g_on_path)
        expand_obstructed(obj_node_id(g), g, parent_on_plan_path=g_on_path)
    lines.append('}')
    return "\n".join(lines)


def draw_and_or_graph(kb, goal, save_path, plan=None, render_png=True):
    """Save an AND-OR graph DOT (and optionally PNG) for `kb` rooted at `goal`.

    Args:
        kb: planner KB.
        goal: obj_id or list of obj_ids.
        save_path: target path; extension is replaced/added as needed. The
                   `.dot` source is always written. PNG is rendered next to it
                   when `render_png=True` and the `dot` binary is on PATH.
        plan: optional list of (obj_id, [grasp_ids]) from `plan_with_grasps`.
              When given, plan obj borders + chosen clause borders are colored
              by step (target=red, others cycle `_STEP_PALETTE`).
        render_png: try to invoke `dot -Tpng` to materialize a bitmap.

    Returns:
        dict with the actual paths written (`dot`, optionally `png`).
    """
    if isinstance(goal, int):
        goal = [goal]
    base, _ = os.path.splitext(save_path)
    dot_path = base + ".dot"
    os.makedirs(os.path.dirname(dot_path) or ".", exist_ok=True)
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(_emit_dot(kb, goal, plan=plan))
    out = {'dot': dot_path}

    if render_png:
        png_path = base + ".png"
        try:
            subprocess.run(
                ["dot", "-Tpng", dot_path, "-o", png_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            out['png'] = png_path
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            stderr = getattr(e, 'stderr', b'') or b''
            print(f"[draw_and_or_graph] PNG render skipped: "
                  f"{type(e).__name__}: {stderr.decode(errors='ignore').strip()}")
    return out
