import math
import copy
import heapq
import random
import time
import statistics as st


# ----------------------------------------------------------------- DATA
COORDS = {
    "A": (16.516104, 107.513540),
    "N": (16.513364, 107.498871),
    "L": (16.539077, 107.472714),
    "M": (16.529061, 107.525547),
    "B": (16.519024, 107.545199),
    "C": (16.568478, 107.514781),
    "D": (16.591244, 107.494793),
    "F": (16.561571, 107.534246),
    "G": (16.574682, 107.584344),
    "E": (16.612000, 107.528000),
    "H": (16.603000, 107.565000),
    "K": (16.545000, 107.575000),
}
BASE_EDGES = [
    ("A", "N", 1.6),
    ("A", "M", 4.2),
    ("A", "B", 3.8),
    ("N", "L", 5.8),
    ("L", "F", 9.5),
    ("L", "K", 13.1),
    ("M", "L", 6.4),
    ("M", "C", 5.0),
    ("B", "C", 4.8),
    ("B", "D", 8.2),
    ("C", "F", 2.0),
    ("D", "F", 5.2),
    ("D", "E", 4.9),
    ("F", "G", 6.2),
    ("F", "K", 5.2),
    ("F", "H", 6.8),
    ("E", "H", 4.6),
    ("H", "G", 4.2),
    ("K", "G", 4.9),
]
START = "A"
CANDIDATE_NODES = [n for n in COORDS if n != START]  # 11 candidate rescue nodes
NUM_TRIALS = 100
RANDOM_SEED = 42


def haversine_distance(c1, c2):
    R = 6371.0
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


HAVERSINE_MATRIX = {
    u: {v: haversine_distance(COORDS[u], COORDS[v]) for v in COORDS} for u in COORDS
}


# ----------------------------------------------------------------- GRAPH
class Graph:
    def __init__(self):
        self.adj = {}

    def add_node(self, n):
        self.adj.setdefault(n, {})

    def add_edge(self, u, v, **attrs):
        self.add_node(u)
        self.add_node(v)
        data = dict(attrs)
        self.adj[u][v] = data
        self.adj[v][u] = data

    def neighbors(self, u):
        return iter(self.adj[u])

    def __getitem__(self, u):
        return self.adj[u]

    def edges(self):
        """Mỗi cạnh vô hướng đúng một lần, theo thứ tự thêm vào."""
        seen, out = set(), []
        for n, nbrs in self.adj.items():
            for nbr in nbrs:
                if nbr not in seen:
                    out.append((n, nbr))
            seen.add(n)
        return out


def build_undirected_graph(coords, edges):
    G = Graph()
    for node in coords:
        G.add_node(node)
    for u, v, d in edges:
        G.add_edge(u, v, base_weight=d, weight=d, penalty=1.0, status="safe")
    return G


def shortest_path_lengths(G, source, weight="base_weight"):
    """Dijkstra một nguồn: trả về {node: khoảng cách ngắn nhất}."""
    dist = {}
    best = {source: 0.0}
    pq = [(0.0, 0, source)]
    counter = 0
    while pq:
        d, _, u = heapq.heappop(pq)
        if u in dist:
            continue
        dist[u] = d
        for v, attrs in G[u].items():
            nd = d + attrs[weight]
            if v not in dist and nd < best.get(v, float("inf")):
                best[v] = nd
                counter += 1
                heapq.heappush(pq, (nd, counter, v))
    return dist


G_base = build_undirected_graph(COORDS, BASE_EDGES)
BASE_SP_DIST = {s: shortest_path_lengths(G_base, s, "base_weight") for s in G_base.adj}


# ------------------------------------------------------------ HEURISTICS
def h_gbfs_state(u, visited, rescue_set):
    remaining = rescue_set - visited
    if not remaining:
        return 0.0
    return min(HAVERSINE_MATRIX[u][r] for r in remaining)


def h_astar_state(u, visited, rescue_set):
    remaining = rescue_set - visited
    if not remaining:
        return 0.0
    return max(BASE_SP_DIST[u][r] for r in remaining)


# --------------------------------------------------------- TRIAL GENERATOR
def apply_random_road_conditions(
    G, rng, num_penalized=10, num_blocked=1, penalty_range=(1.0, 1.5)
):
    G_trial = copy.deepcopy(G)
    all_edges = list(G_trial.edges())
    rng.shuffle(all_edges)
    blocked = all_edges[:num_blocked]
    penalized = all_edges[num_blocked : num_blocked + num_penalized]
    for u, v in blocked:
        G_trial[u][v]["status"] = "blocked"
        G_trial[u][v]["penalty"] = float("inf")
        G_trial[u][v]["weight"] = float("inf")
    for u, v in penalized:
        f = rng.uniform(*penalty_range)
        G_trial[u][v]["status"] = "penalized"
        G_trial[u][v]["penalty"] = f
        G_trial[u][v]["weight"] = G_trial[u][v]["base_weight"] * f
    return G_trial, penalized, blocked


def generate_trials(n=NUM_TRIALS, seed=RANDOM_SEED):
    """One shared Random(seed): rescue sampling -> edge shuffle -> penalty sampling."""
    rng = random.Random(seed)
    trials = []
    for _ in range(n):
        rescue = rng.sample(CANDIDATE_NODES, 3)
        G_trial, pen, blk = apply_random_road_conditions(G_base, rng)
        trials.append((rescue, G_trial, pen, blk))
    return trials


# ------------------------------------------------------------- BENCHMARK
def dijkstra_state_space(G, start, rescue_locations):
    """Optimal cost on augmented state space (h = 0)."""
    rescue_set = frozenset(rescue_locations)
    init = frozenset([start]) & rescue_set
    pq = [(0.0, 0, start, init)]
    best = {(start, init): 0.0}
    counter = 0
    while pq:
        g, _, u, vis = heapq.heappop(pq)
        if g > best.get((u, vis), float("inf")):
            continue
        if vis == rescue_set:
            return g
        for v in G.neighbors(u):
            w = G[u][v]["weight"]
            if not math.isfinite(w):
                continue
            nv = vis | (frozenset([v]) & rescue_set)
            ng = g + w
            if ng < best.get((v, nv), float("inf")):
                best[(v, nv)] = ng
                counter += 1
                heapq.heappush(pq, (ng, counter, v, nv))
    return float("inf")


def route_costs(G, path):
    """Return (penalized_cost, base_cost); raises if a blocked edge is used."""
    pen = base = 0.0
    for u, v in zip(path[:-1], path[1:]):
        w = G[u][v]["weight"]
        if not math.isfinite(w):
            raise ValueError(f"Route uses blocked edge {u}-{v}")
        pen += w
        base += G[u][v]["base_weight"]
    return pen, base


def is_valid_route(G, path, rescue_locations):
    if path is None or path[0] != START:
        return False
    try:
        route_costs(G, path)
    except ValueError:
        return False
    return set(rescue_locations) <= set(path)


def is_optimal(cost, dijkstra_cost):
    return math.isclose(cost, dijkstra_cost, rel_tol=1e-9, abs_tol=1e-9)


"""Global Greedy Best-First Search"""


def global_gbfs_search(G, start, rescue_locations):
    rescue_set = frozenset(rescue_locations)
    init_visited = frozenset([start]) & rescue_set
    h0 = h_gbfs_state(start, init_visited, rescue_set)
    counter = 0
    pq = [(h0, counter, start, init_visited, [start], 0.0)]
    visited_states = set()
    expanded = 0
    while pq:
        _, _, u, vis, path, g = heapq.heappop(pq)
        state = (u, vis)
        if state in visited_states:
            continue
        visited_states.add(state)
        expanded += 1
        if vis == rescue_set:
            return path, g, expanded
        for v in G.neighbors(u):
            w = G[u][v]["weight"]
            if not math.isfinite(w):
                continue
            nvis = vis | (frozenset([v]) & rescue_set)
            if (v, nvis) not in visited_states:
                counter += 1
                heapq.heappush(
                    pq,
                    (
                        h_gbfs_state(v, nvis, rescue_set),
                        counter,
                        v,
                        nvis,
                        path + [v],
                        g + w,
                    ),
                )
    return None, float("inf"), expanded


def show_trial(trial_index=0):
    rescue, G_trial, pen, blk = generate_trials(trial_index + 1)[trial_index]
    t0 = time.perf_counter()
    path, cost, expanded = global_gbfs_search(G_trial, START, rescue)
    ms = (time.perf_counter() - t0) * 1000
    pen_cost, base_cost = route_costs(G_trial, path)
    opt = dijkstra_state_space(G_trial, START, rescue)
    print(f"=== Global GBFS | Trial {trial_index + 1} ===")
    print(f"Rescue set      : {sorted(rescue)}")
    print(f"Blocked edge    : {blk[0][0]}-{blk[0][1]}")
    print(f"Route           : {' -> '.join(path)}")
    print(f"Penalized cost  : {pen_cost:.3f} km")
    print(f"Base cost       : {base_cost:.3f} km")
    print(f"Expanded units  : {expanded}")
    print(f"Time            : {ms:.4f} ms")
    print(f"Valid route     : {is_valid_route(G_trial, path, rescue)}")
    print(f"Dijkstra optimum: {opt:.3f} km | matches: {is_optimal(pen_cost, opt)}")


if __name__ == "__main__":
    show_trial(0)

"""Global A* on augmented state space"""


def global_a_star_search(G_trial, start, rescue_locations, base_sp_dist=BASE_SP_DIST):
    rescue_set = frozenset(rescue_locations)
    init_visited = frozenset([start]) & rescue_set

    def h(u, visited):
        remaining = rescue_set - visited
        if not remaining:
            return 0.0
        return max(base_sp_dist[u][r] for r in remaining)

    counter = 0
    open_set = [(h(start, init_visited), 0.0, counter, start, init_visited, [start])]
    best_g = {(start, init_visited): 0.0}
    expanded = 0
    while open_set:
        _, g, _, u, visited, path = heapq.heappop(open_set)
        if g > best_g.get((u, visited), float("inf")):
            continue
        expanded += 1
        if visited == rescue_set:
            return path, g, expanded
        for v in G_trial.neighbors(u):
            w = G_trial[u][v]["weight"]
            if not math.isfinite(w):
                continue
            nvis = visited | (frozenset([v]) & rescue_set)
            tg = g + w
            if tg < best_g.get((v, nvis), float("inf")):
                best_g[(v, nvis)] = tg
                counter += 1
                heapq.heappush(
                    open_set, (tg + h(v, nvis), tg, counter, v, nvis, path + [v])
                )
    return None, float("inf"), expanded


def show_trial(trial_index=0):
    rescue, G_trial, pen, blk = generate_trials(trial_index + 1)[trial_index]
    t0 = time.perf_counter()
    path, cost, expanded = global_a_star_search(G_trial, START, rescue)
    ms = (time.perf_counter() - t0) * 1000
    pen_cost, base_cost = route_costs(G_trial, path)
    opt = dijkstra_state_space(G_trial, START, rescue)
    print(f"=== Global A* | Trial {trial_index + 1} ===")
    print(f"Rescue set      : {sorted(rescue)}")
    print(f"Blocked edge    : {blk[0][0]}-{blk[0][1]}")
    print(f"Route           : {' -> '.join(path)}")
    print(f"Penalized cost  : {pen_cost:.3f} km")
    print(f"Base cost       : {base_cost:.3f} km")
    print(f"Expanded units  : {expanded}")
    print(f"Time            : {ms:.4f} ms")
    print(f"Valid route     : {is_valid_route(G_trial, path, rescue)}")
    print(f"Dijkstra optimum: {opt:.3f} km | matches: {is_optimal(pen_cost, opt)}")


if __name__ == "__main__":
    show_trial(0)

"""Greedy (nearest-neighbour by Haversine) + leg-by-leg A*"""


def single_leg_a_star(G_trial, start_node, target_node, base_sp_dist=BASE_SP_DIST):
    counter = 0
    open_set = [
        (base_sp_dist[start_node][target_node], 0.0, counter, start_node, [start_node])
    ]
    best_g = {start_node: 0.0}
    expanded = 0
    while open_set:
        _, g, _, u, path = heapq.heappop(open_set)
        if g > best_g.get(u, float("inf")):
            continue
        expanded += 1
        if u == target_node:
            return path, g, expanded
        for v in G_trial.neighbors(u):
            w = G_trial[u][v]["weight"]
            if not math.isfinite(w):
                continue
            tg = g + w
            if tg < best_g.get(v, float("inf")):
                best_g[v] = tg
                counter += 1
                heapq.heappush(
                    open_set,
                    (tg + base_sp_dist[v][target_node], tg, counter, v, path + [v]),
                )
    return None, float("inf"), expanded


def greedy_plus_a_star_search(
    G_trial, start, rescue_locations, base_sp_dist=BASE_SP_DIST
):
    remaining = set(rescue_locations) - {start}
    curr = start
    full_path = [start]
    total_cost = 0.0
    total_expanded = 0
    while remaining:
        next_target = min(remaining, key=lambda r: HAVERSINE_MATRIX[curr][r])  # Stage 1
        leg_path, leg_cost, leg_exp = single_leg_a_star(
            G_trial, curr, next_target, base_sp_dist
        )  # Stage 2
        total_expanded += leg_exp
        if leg_path is None:
            return None, float("inf"), total_expanded
        full_path.extend(leg_path[1:])
        total_cost += leg_cost
        for node in leg_path:
            remaining.discard(node)
        curr = next_target
    return full_path, total_cost, total_expanded


def show_trial(trial_index=0):
    rescue, G_trial, pen, blk = generate_trials(trial_index + 1)[trial_index]
    t0 = time.perf_counter()
    path, cost, expanded = greedy_plus_a_star_search(G_trial, START, rescue)
    ms = (time.perf_counter() - t0) * 1000
    pen_cost, base_cost = route_costs(G_trial, path)
    opt = dijkstra_state_space(G_trial, START, rescue)
    print(f"=== Greedy + A* | Trial {trial_index + 1} ===")
    print(f"Rescue set      : {sorted(rescue)}")
    print(f"Blocked edge    : {blk[0][0]}-{blk[0][1]}")
    print(f"Route           : {' -> '.join(path)}")
    print(f"Penalized cost  : {pen_cost:.3f} km")
    print(f"Base cost       : {base_cost:.3f} km")
    print(f"Expanded units  : {expanded}")
    print(f"Time            : {ms:.4f} ms")
    print(f"Valid route     : {is_valid_route(G_trial, path, rescue)}")
    print(f"Dijkstra optimum: {opt:.3f} km | matches: {is_optimal(pen_cost, opt)}")


if __name__ == "__main__":
    show_trial(0)

"""Run 100 seeded trials, compare 3 algorithms"""

ALGOS = {
    "GBFS": lambda G, s, r: global_gbfs_search(G, s, r),
    "Global A*": lambda G, s, r: global_a_star_search(G, s, r, BASE_SP_DIST),
    "Greedy + A*": lambda G, s, r: greedy_plus_a_star_search(G, s, r, BASE_SP_DIST),
}


def main():
    trials = generate_trials(NUM_TRIALS, RANDOM_SEED)
    res = {
        n: {"pen": [], "base": [], "exp": [], "ms": [], "ok": 0, "opt": 0}
        for n in ALGOS
    }

    for rescue, G_trial, _, _ in trials:
        opt = dijkstra_state_space(G_trial, START, rescue)
        for name, fn in ALGOS.items():
            t0 = time.perf_counter()
            path, cost, expanded = fn(G_trial, START, rescue)
            ms = (time.perf_counter() - t0) * 1000
            if not is_valid_route(G_trial, path, rescue):
                continue
            pen, base = route_costs(G_trial, path)
            r = res[name]
            r["ok"] += 1
            r["pen"].append(pen)
            r["base"].append(base)
            r["exp"].append(expanded)
            r["ms"].append(ms)
            r["opt"] += is_optimal(pen, opt)

    sd = st.stdev  # sample SD; change to st.pstdev for population SD
    print("Table 3: Performance over %d seeded trials" % NUM_TRIALS)
    print(
        f"{'Algorithm':<12}{'Penalized cost (km)':<28}{'Base cost (km)':<28}"
        f"{'Expanded units':<18}{'Time (ms)':<20}{'Success/optimal'}"
    )
    for name, r in res.items():
        print(
            f"{name:<12}"
            f"{st.mean(r['pen']):.3f} ± {sd(r['pen']):.3f} ({max(r['pen']):.3f})".ljust(
                40
            )
            + f"{st.mean(r['base']):.3f} ± {sd(r['base']):.3f} ({max(r['base']):.3f})".ljust(
                28
            )
            + f"{st.mean(r['exp']):.2f} ± {sd(r['exp']):.2f}".ljust(18)
            + f"{st.mean(r['ms']):.4f} ± {sd(r['ms']):.4f}".ljust(20)
            + f"{100 * r['ok'] / NUM_TRIALS:.0f}% / {100 * r['opt'] / NUM_TRIALS:.0f}%"
        )


if __name__ == "__main__":
    main()
