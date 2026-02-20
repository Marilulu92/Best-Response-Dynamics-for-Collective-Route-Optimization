# 🚦 Best-Response-Dynamics-for-Collective-Route-Optimization

**Manuscript:** IEEE Latin America Transactions  
**Paper:** *Best Response Dynamics for Collective Route Optimization*

**Authors**  
Maria de Lourdes Angulo-Dominguez  
Pedro Mejía-Alvarez  
Rolando Menchaca-Mendez  
Arturo Yee-Rendon  

📩 For questions or replication of results: lourdes.angulo@cinvestav.mx  

---

# 📌 Project Overview

This repository contains the official experimental implementation of:

**Best Response Dynamics for Collective Route Optimization**

It implements a Collective Optimization Scheme (COS) for urban routing integrating:

- Best Response Dynamics (BRD)
- Dijkstra shortest-path routing
- Congestion-aware cost model
- Nash equilibrium search
- SUMO simulation integration

The framework generates:

- Individual routing (selfish baseline)
- Collaborative routing (BRD equilibrium)
- Multiple congestion scenarios (10%–100%)
- Table III results (IEEE LATAM)
- SUMO route files for visualization

---

# 🧰 Requirements

## Software

- Python 3.9+
- SUMO Simulator
- netconvert (included in SUMO)

## Verify installation

Run in terminal:
```bash
netconvert --help  
sumo --help  
sumo-gui --help  
```
If not recognized → add SUMO /bin to PATH.

---

# 🗺️ Map Preparation (OpenStreetMap → SUMO)

## Step 1 — Download map

Download from:  
https://www.openstreetmap.org  

Export as:

map.osm

## Step 2 — Convert to SUMO network

Run:
```bash
netconvert --osm-files map.osm -o map.net.xml
```
This generates:

map.net.xml

Required for:
- running algorithm
- SUMO simulation

---

# 🗂️ Folder Structure (per map)

experiments/  
 ├── map_firstmap/  
 │    ├── cos_integrado.py  
 │    ├── map.osm  
 │    ├── map.net.xml  
 │    ├── od_pairs.txt  
 │    └── results/  
 ├── map_secondmap/  
 │    ├── cos_integrado.py  
 │    ├── map.osm
 │    ├── map.net.xml  
 │    ├── od_pairs.txt  
 │    └── results/  
 └──   

---

# 📍 OD Pairs File

File: od_pairs.txt  

Format:

origin_junction destination_junction

Example:
```bash
2746068817 2745809412  
7286566917 1795001889  
```
These IDs correspond to SUMO junctions.

---

# 🔎 How to Obtain Junction IDs

Create file view_map.sumo.cfg with:
```bash
<configuration>
  <input>
    <net-file value="map.net.xml"/>
  </input>
</configuration>
```
Run:
```bash
sumo-gui view_map.sumo.cfg
```
Then:
View → Junctions → Show IDs  
Copy IDs into od_pairs.txt

---

# 🚀 Run Algorithm

python cos.py --net [map.net.xml] --od [od_pairs.txt] --veh_per_od [1] --out [results] --write_routes

---
## ⚙️ Parameters

| Parameter        | Description                              | 
|------------------|------------------------------------------|
| `--net`          | SUMO network file                        |
| `--od`           | OD pairs file                            |
| `--veh_per_od`   | vehicles per OD pair                     |
| `--out`          | output folder                            |
| `--write_routes` | generate SUMO routes                     |
| `--table_pcts`   | congestion levels                        |
| `--max_iters`    | max BRD iterations (optional)            |
| `--seed`         | reproducibility seed (optional)          |

---
Example:
```bash
python cos.py --net map.net.xml --od od_pairs.txt --veh_per_od 1 --out results --write_routes
```
# 📊 Outputs

results/  
 ├── tableIII.csv  
 ├── tableIII.tex  
 ├── run_log.txt  
 └── routes/  
     ├── pct_10/  
     ├── pct_20/  
     └── pct_100/  

Each folder contains:

routes_brd.rou.xml  
routes_dijkstra.rou.xml  

---

# 🚦 SUMO Simulation

Inside each folder pct_XX copy:

map.net.xml  
mapB.sumo.cfg  
mapD.sumo.cfg  

Run:
```bash
sumo-gui mapB.sumo.cfg  
sumo-gui mapD.sumo.cfg  
```
---

# 📚 Research Context

Repository supporting:  
Best Response Dynamics for Collective Route Optimization  
IEEE Latin America Transactions

---

# 📩 Contact

lourdes.angulo@cinvestav.mx
