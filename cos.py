#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COS / BRD pipeline + TABLE III generator (NO SUMO simulation, but writes SUMO route files)

What this script does:
- (Optional) Convert OSM -> NETXML with netconvert
- Parse *.net.xml, build graph
- Sweep multiple congestion percentages (default 10..100 step 10)
- For each congestion level:
    * Individual routing (Dijkstra init) total travel time (divided by 60)
    * Cooperative routing (BRD) total travel time (divided by 60)
    * Diff (%) improvement
    * BRD (I): BRD rounds until Nash (a full round with no changes)
    * Vehicles (≈): background vehicles from congestion + OD vehicles (players)
- Write:
    out_dir/tableIII.csv
    out_dir/tableIII.tex   (IEEE-ready tabular)
    out_dir/run_log.txt
    and compat files (Open.txt, carriles.txt, ...)

NEW (as requested):
- If you pass --write_routes, it will write route files for EACH scenario (each congestion pct):
    out_dir/routes/pct_10/routes_dijkstra.rou.xml
    out_dir/routes/pct_10/routes_brd.rou.xml
    ...
    out_dir/routes/pct_100/...

Demand control:
- --veh_per_od N replicates each OD pair N times, so each OD represents N vehicles (players).

BRD order:
- FIXED 0..N-1 (no shuffle).
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TextIO

import xml.etree.ElementTree as ET


# ----------------------------
# Tee logger (stdout + file)
# ----------------------------

class TeeLogger:
    def __init__(self, file: TextIO):
        self.file = file

    def print(self, *args, sep: str = " ", end: str = "\n") -> None:
        msg = sep.join(str(a) for a in args) + end
        # console
        try:
            import sys
            sys.stdout.write(msg)
            sys.stdout.flush()
        except Exception:
            pass
        # file
        self.file.write(msg)
        self.file.flush()


# ----------------------------
# OSM -> NETXML
# ----------------------------

def netconvert_osm_to_netxml(osm_file: Path, netxml_out: Path) -> None:
    """
    Convert an OpenStreetMap .osm file to a SUMO network .net.xml using netconvert.

    Requires: netconvert available in PATH.
    Example:
        netconvert --osm-files map.osm -o map.net.xml
    """
    import subprocess

    if not osm_file.exists():
        raise FileNotFoundError(f"No existe el archivo OSM: {osm_file}")

    netxml_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["netconvert", "--osm-files", str(osm_file), "-o", str(netxml_out)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "No encuentro 'netconvert' en tu PATH. "
            "Asegúrate de tener SUMO instalado y netconvert disponible (o añade SUMO/bin a PATH)."
        ) from e


# ----------------------------
# Data model
# ----------------------------

@dataclass
class EdgeInfo:
    edge_id: str
    from_node: str
    to_node: str
    speed: float
    length: float
    lanes: int


# Graph:
# grafo[u][v] = [edge_id, speed, length, lanes, cars, weight]
Graph = Dict[str, Dict[str, List[float | int | str]]]


# ----------------------------
# XML parsing
# ----------------------------

def parse_sumo_net_xml(net_xml: Path, *, ignore_internal: bool = True) -> List[EdgeInfo]:
    tree = ET.parse(net_xml)
    root = tree.getroot()

    edges: List[EdgeInfo] = []

    for e in root.findall("edge"):
        edge_id = e.get("id", "")
        function = e.get("function", "")

        if ignore_internal and (edge_id.startswith(":") or function == "internal"):
            continue

        frm = e.get("from")
        to = e.get("to")
        if not frm or not to:
            continue

        lanes = e.findall("lane")
        if not lanes:
            continue

        lane_speeds: List[float] = []
        lane_lengths: List[float] = []
        for ln in lanes:
            s = ln.get("speed")
            l = ln.get("length")
            if s is not None:
                lane_speeds.append(float(s))
            if l is not None:
                lane_lengths.append(float(l))

        speed = max(lane_speeds) if lane_speeds else 1.0
        length = max(lane_lengths) if lane_lengths else 1.0

        edges.append(
            EdgeInfo(
                edge_id=edge_id,
                from_node=frm,
                to_node=to,
                speed=speed,
                length=length,
                lanes=len(lanes),
            )
        )

    return edges


def write_compat_files(edges: List[EdgeInfo], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "Open.txt").write_text(
        "\n".join(f"{e.edge_id}\t{e.speed}\t{e.length}" for e in edges) + "\n",
        encoding="utf-8",
    )
    (out_dir / "carriles.txt").write_text(
        "\n".join(str(e.lanes) for e in edges) + "\n",
        encoding="utf-8",
    )
    (out_dir / "aristasCDMX.txt").write_text(
        "\n".join(f"{e.edge_id}\t{e.from_node}\t{e.to_node}" for e in edges) + "\n",
        encoding="utf-8",
    )
    (out_dir / "OrigenCDMX.txt").write_text(
        "\n".join(e.from_node for e in edges) + "\n", encoding="utf-8"
    )
    (out_dir / "DestinoCDMX.txt").write_text(
        "\n".join(e.to_node for e in edges) + "\n", encoding="utf-8"
    )
    (out_dir / "ConeccionesCDMX.txt").write_text(
        "\n".join(e.edge_id for e in edges) + "\n",
        encoding="utf-8",
    )

    nodes = sorted({e.from_node for e in edges} | {e.to_node for e in edges})
    (out_dir / "NodosCDMX.txt").write_text("\n".join(nodes) + "\n", encoding="utf-8")

    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for e in edges:
        adj[e.from_node].append(e.to_node)

    lines: List[str] = []
    for n in nodes:
        lines.append(n)
        lines.extend(adj[n])
        lines.append("")
    (out_dir / "GraficarCDMX.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------
# Congestion model + Graph
# ----------------------------

def congestion_sigmoid(
    length: float,
    v_max: float,
    lanes: int,
    cars: float,
    *,
    v_min: float = 1.0,
    veh_len: float = 5.0,
    M: float = 10.0,
) -> float:
    carga_max = (length / veh_len) * lanes
    k1 = length / v_min - length / v_max
    k4 = length / v_max
    k3 = carga_max / 2.0
    k2 = (M + k3) / carga_max if carga_max > 0 else 1.0
    return float(k1 / (1.0 + math.exp(-(k2 * cars - k3))) + k4)


def build_graph(edges: List[EdgeInfo], *, initial_load_factor: float) -> Graph:
    grafo: Graph = {}
    for e in edges:
        grafo.setdefault(e.from_node, {})
        capacity = (e.length / 5.0) * e.lanes
        cars = capacity * float(initial_load_factor)
        weight = congestion_sigmoid(e.length, e.speed, e.lanes, cars)
        grafo[e.from_node][e.to_node] = [
            e.edge_id,
            float(e.speed),
            float(e.length),
            int(e.lanes),
            float(cars),
            float(weight),
        ]
    return grafo


def congestion_add(grafo: Graph, route: List[str]) -> None:
    for u, v in zip(route, route[1:]):
        edge = grafo[u][v]
        edge[4] = float(edge[4]) + 1.0
        edge[5] = congestion_sigmoid(float(edge[2]), float(edge[1]), int(edge[3]), float(edge[4]))


def congestion_remove(grafo: Graph, route: List[str]) -> None:
    for u, v in zip(route, route[1:]):
        edge = grafo[u][v]
        edge[4] = max(0.0, float(edge[4]) - 1.0)
        edge[5] = congestion_sigmoid(float(edge[2]), float(edge[1]), int(edge[3]), float(edge[4]))


def route_cost(grafo: Graph, route: List[str]) -> float:
    return sum(float(grafo[u][v][5]) for u, v in zip(route, route[1:]))


# ----------------------------
# Dijkstra
# ----------------------------

def dijkstra_shortest_path(grafo: Graph, src: str, dst: str) -> List[str]:
    import heapq

    if src not in grafo:
        raise KeyError(f"Origen desconocido: {src}")
    if src == dst:
        return [src]

    dist: Dict[str, float] = {src: 0.0}
    prev: Dict[str, Optional[str]] = {src: None}
    pq: List[Tuple[float, str]] = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d != dist.get(u, float("inf")):
            continue

        for v in grafo.get(u, {}):
            w = float(grafo[u][v][5])
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dst not in prev:
        raise ValueError(f"No hay ruta entre {src} y {dst}")

    path: List[str] = []
    cur: Optional[str] = dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


# ----------------------------
# Route XML writer (for SUMO input)
# ----------------------------

def write_routes_xml(grafo: Graph, routes: List[List[str]], out_file: Path, *, depart: float = 10.0) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<routes>\n")

        for vid, route_nodes in enumerate(routes):
            edges: List[str] = []
            for u, v in zip(route_nodes, route_nodes[1:]):
                edges.append(str(grafo[u][v][0]))

            f.write(f'  <vehicle id="{vid}" depart="{depart:.2f}">\n')
            f.write(f'    <route edges="{" ".join(edges)}"/>\n')
            f.write("  </vehicle>\n")

        f.write("</routes>\n")


# ----------------------------
# BRD (fixed order)
# ----------------------------

def brd_equilibrium(
    grafo: Graph,
    od_pairs: List[Tuple[str, str]],
    *,
    max_iters: int = 2000,
    stop_when_full_round_no_change: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[List[str]], List[float], int, int, int]:
    """
    Returns:
      routes, costs, changes, rounds_done, total_evaluations
    """
    if seed is not None:
        random.seed(seed)

    routes: List[List[str]] = [dijkstra_shortest_path(grafo, o, d) for (o, d) in od_pairs]
    for r in routes:
        congestion_add(grafo, r)

    costs = [route_cost(grafo, r) for r in routes]

    num_changes = 0
    idxs = list(range(len(routes)))  # FIXED order

    i = 0
    total_evaluations = 0

    while i < max_iters:
        i += 1
        changed_this_round = False

        for j in idxs:
            total_evaluations += 1
            o, d = od_pairs[j]
            old_cost = costs[j]
            old_route = routes[j]

            new_route = dijkstra_shortest_path(grafo, o, d)
            new_cost = route_cost(grafo, new_route)

            if new_cost < old_cost:
                congestion_add(grafo, new_route)
                congestion_remove(grafo, old_route)
                routes[j] = new_route
                costs[j] = new_cost
                num_changes += 1
                changed_this_round = True

        if stop_when_full_round_no_change and not changed_this_round:
            break

    rounds_done = i
    return routes, costs, num_changes, rounds_done, total_evaluations


# ----------------------------
# OD file
# ----------------------------

def parse_od_file(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pairs.append((parts[0], parts[1]))
    return pairs


def parse_pct_list(s: str) -> List[float]:
    vals: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


# ----------------------------
# Table writer
# ----------------------------

def write_tableIII(out_dir: Path, rows: List[dict]) -> None:
    """
    Writes:
      tableIII.csv
      tableIII.tex
    Vehicles (≈) = background + OD vehicles
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "tableIII.csv"
    header = ["Indv (T)", "Colab (T)", "Diff (%)", "BRD (I)", "% Congestion", "Vehicles (≈)"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(
            f"{r['indv']:.2f},{r['colab']:.2f},{r['diff']:.2f},{r['brd_i']},{r['pct']:.0f},≈ {r['veh']:,}"
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # LaTeX
    tex_path = out_dir / "tableIII.tex"
    tex_lines: List[str] = []
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\caption{Travel times for individual vs. cooperative routing.}")
    tex_lines.append(r"\label{tab:tableIII}")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\begin{tabular}{r r r r r r}")
    tex_lines.append(r"\hline")
    tex_lines.append(r"Indv (T) & Colab (T) & Diff (\%) & BRD (I) & \% Congestion & Vehicles ($\approx$) \\")
    tex_lines.append(r"\hline")
    for r in rows:
        # IMPORTANT: avoid tricky escaping -> use format()
        tex_lines.append(
            "{} & {} & {} & {} & {} & $\\approx$ {} \\\\".format(
                f"{r['indv']:.2f}",
                f"{r['colab']:.2f}",
                f"{r['diff']:.2f}",
                r["brd_i"],
                f"{r['pct']:.0f}",
                format(r["veh"], ","),
            )
        )
    tex_lines.append(r"\hline")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


# ----------------------------
# Core experiment for one congestion level
# ----------------------------

def run_one_level(
    *,
    edges: List[EdgeInfo],
    od_pairs: List[Tuple[str, str]],
    congestion_pct: float,
    max_iters: int,
    seed: int,
) -> dict:
    load_factor = congestion_pct / 100.0

    # Individual routing (Dijkstra init)
    grafo_d = build_graph(edges, initial_load_factor=load_factor)
    routes_dij = [dijkstra_shortest_path(grafo_d, o, d) for (o, d) in od_pairs]
    for r in routes_dij:
        congestion_add(grafo_d, r)
    costs_dij = [route_cost(grafo_d, r) for r in routes_dij]
    indv = sum(costs_dij) / 60.0

    # Cooperative routing (BRD)
    grafo_b = build_graph(edges, initial_load_factor=load_factor)
    routes_brd, costs_brd, changes, rounds, evals = brd_equilibrium(
        grafo_b,
        od_pairs,
        max_iters=max_iters,
        seed=seed,
    )
    colab = sum(costs_brd) / 60.0

    diff = (1.0 - (colab / indv)) * 100.0 if indv > 0 else 0.0

    # Vehicles for table: background + OD vehicles
    total_capacity = sum((e.length / 5.0) * e.lanes for e in edges)
    vehicles_bg = int(round(total_capacity * load_factor))
    vehicles_od = len(od_pairs)
    vehicles = vehicles_bg + vehicles_od

    return {
        "pct": congestion_pct,
        "veh": vehicles,
        "veh_bg": vehicles_bg,
        "veh_od": vehicles_od,
        "indv": indv,
        "colab": colab,
        "diff": diff,
        "brd_i": rounds,
        "brd_changes": changes,
        "brd_evals": evals,
        # for optional routes writing:
        "routes_dij": routes_dij,
        "routes_brd": routes_brd,
        "grafo_d": grafo_d,
        "grafo_b": grafo_b,
    }


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="COS/BRD pipeline. Generates Table III (multiple congestion levels)."
    )

    ap.add_argument("--net", type=Path, help="Path to SUMO network XML (*.net.xml)")
    ap.add_argument("--osm", type=Path, help="Optional: OSM file (map.osm) to convert with netconvert")
    ap.add_argument("--net_out", type=Path, default=Path("map.net.xml"), help="Output .net.xml path when using --osm")

    ap.add_argument("--out", default=Path("out_cmx"), type=Path, help="Output directory")
    ap.add_argument("--od", type=Path, required=True, help="File with OD pairs (origin destination per line)")
    ap.add_argument("--veh_per_od", type=int, default=1, help="Vehículos por OD pair (replica cada OD este número de veces).")

    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max_iters", type=int, default=2000, help="Max BRD rounds")

    ap.add_argument(
        "--table_pcts",
        type=str,
        default="10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated congestion percentages to evaluate (default 10..100 step 10).",
    )

    # NEW: write routes for EACH scenario
    ap.add_argument(
        "--write_routes",
        action="store_true",
        help="Write routes for EACH congestion scenario under out/routes/pct_XX/ (Dijkstra + BRD).",
    )

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "run_log.txt"

    with log_path.open("w", encoding="utf-8") as lf:
        log = TeeLogger(lf)

        # Resolve network input
        if args.net:
            net_path = args.net
        elif args.osm:
            net_path = args.net_out
            log.print("[INFO] Convirtiendo OSM -> NETXML con netconvert...")
            netconvert_osm_to_netxml(args.osm, net_path)
        else:
            log.print("[ERROR] Debes proporcionar --net o --osm.")
            return 2

        log.print(f"[INFO] Network file: {net_path}")

        edges = parse_sumo_net_xml(net_path, ignore_internal=True)
        if not edges:
            log.print("[ERROR] No edges parsed. Revisa que --net apunte a un archivo válido (*.net.xml).")
            return 2
        log.print(f"[INFO] Edges parsed: {len(edges)}")

        # Write compat files once
        write_compat_files(edges, args.out)
        log.print(f"[INFO] Wrote compat files to: {args.out}")

        # Load and expand OD pairs (veh_per_od)
        od_pairs_base = parse_od_file(args.od)
        log.print(f"[INFO] OD file: {args.od} (pairs={len(od_pairs_base)})")

        if args.veh_per_od < 1:
            log.print("[ERROR] --veh_per_od debe ser >= 1")
            return 2

        od_pairs: List[Tuple[str, str]] = []
        for (o, d) in od_pairs_base:
            for _ in range(args.veh_per_od):
                od_pairs.append((o, d))
        log.print(f"[INFO] Vehicles per OD: {args.veh_per_od} -> total OD vehicles(players)={len(od_pairs)}")

        pcts = parse_pct_list(args.table_pcts)
        rows_simple: List[dict] = []

        log.print(f"[INFO] Running TABLE III sweep for pcts={pcts} ...")

        routes_root = args.out / "routes"
        if args.write_routes:
            routes_root.mkdir(parents=True, exist_ok=True)

        for pct in pcts:
            log.print(f"[INFO]  -> running pct={pct:.0f}%")
            row = run_one_level(
                edges=edges,
                od_pairs=od_pairs,
                congestion_pct=float(pct),
                max_iters=args.max_iters,
                seed=args.seed,
            )

            rows_simple.append(
                {k: row[k] for k in ["pct", "veh", "veh_bg", "veh_od", "indv", "colab", "diff", "brd_i", "brd_changes", "brd_evals"]}
            )

            log.print(
                "[INFO]     done pct={} | Indv={:.2f} Colab={:.2f} Diff={:.2f}% BRD(I)={} Vehicles≈{}".format(
                    int(round(pct)),
                    row["indv"],
                    row["colab"],
                    row["diff"],
                    row["brd_i"],
                    format(row["veh"], ","),
                )
            )

            # NEW: write routes for each scenario
            if args.write_routes:
                pct_dir = routes_root / f"pct_{int(round(pct))}"
                pct_dir.mkdir(parents=True, exist_ok=True)

                write_routes_xml(row["grafo_d"], row["routes_dij"], pct_dir / "routes_dijkstra.rou.xml")
                write_routes_xml(row["grafo_b"], row["routes_brd"], pct_dir / "routes_brd.rou.xml")

                log.print(f"[INFO]     wrote routes to: {pct_dir}")

        # Write table outputs (csv + tex)
        write_tableIII(args.out, rows_simple)
        log.print(f"[INFO] Wrote table outputs: {args.out/'tableIII.csv'} and {args.out/'tableIII.tex'}")

        log.print(f"[INFO] Full run log saved to: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
