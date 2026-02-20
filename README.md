🚦 Best-Response-Dynamics-for-Collective-Route-Optimization

IEEE LATAM Q1 Repository (Official Experimental Implementation)

Manuscript: IEEE Latin America Transactions
Paper: Best Response Dynamics for Collective Route Optimization

Authors
Maria de Lourdes Angulo-Dominguez
Pedro Mejía-Alvarez
Rolando Menchaca-Mendez
Arturo Yee-Rendon

For questions or replication of results:
📩 lourdes.angulo@cinvestav.mx

📌 Project Overview

This repository contains the official experimental implementation of the research paper:

Best Response Dynamics for Collective Route Optimization

The system implements a Collective Optimization Scheme (COS) for urban routing based on:

Best Response Dynamics (BRD)

Dijkstra shortest-path routing

Congestion-aware cost model

Nash equilibrium search

SUMO simulation integration

The framework generates:

Individual routing (selfish/Dijkstra)

Collaborative routing (BRD equilibrium)

Congestion scenarios (10%–100%)

Table III results for IEEE LATAM

Route files ready for SUMO simulation

🧰 Requirements
Software

Python 3.9+

SUMO simulator

netconvert (included with SUMO)

Verify installation
netconvert --help
sumo --help
sumo-gui --help

If not recognized → add SUMO /bin to PATH.

🗺️ Map Preparation (OpenStreetMap → SUMO)
Step 1 — Download map

Download from OpenStreetMap:

https://www.openstreetmap.org

Export as:

map.osm
Step 2 — Convert to SUMO network
netconvert --osm-files map.osm -o map.net.xml

This generates:

map.net.xml

Required for:

algorithm

SUMO simulation

🗂️ Recommended Folder Structure (per map)

Each map must have its own folder:

experiments/
 ├── map_gdl/
 │    ├── map.osm
 │    ├── map.net.xml
 │    ├── od_pairs.txt
 │    └── results/
 ├── map_cdmx/
 └── map_mty/
📍 OD Pairs File

File:

od_pairs.txt

Format:

origin_junction destination_junction

Example:

2746068817 2745809412
7286566917 1795001889

These IDs correspond to junctions from the SUMO map.

🔎 How to Obtain Junction IDs (Manual Method)

Create:

view_map.sumo.cfg
<configuration>
  <input>
    <net-file value="map.net.xml"/>
  </input>
</configuration>

Run:

sumo-gui view_map.sumo.cfg

Then:

View → Junctions

Show IDs

Copy IDs into od_pairs.txt

(This is currently manual for this research stage.)

🚀 Run Algorithm (ONE LINE COMMAND)
python cos_integrado_tableIII_veh_per_od_routes_all.py --net map.net.xml --od od_pairs.txt --veh_per_od 1 --out results --write_routes
⚙️ Parameters
Parameter	Meaning
--net	SUMO network file
--od	OD pairs file
--veh_per_od	vehicles per OD pair
--out	output folder
--write_routes	generate SUMO routes
--table_pcts	congestion levels
--max_iters (Optional)	BRD max iterations
--seed (Optional)	reproducibility

Example:

 python cos_integrado_tableIII_veh_per_od_routes_all.py --net map.net.xml --od od_pairs.txt --veh_per_od 5 --out results --write_routes  

means 5 vehicles per OD pair.

📊 Outputs Generated

Inside results:

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
🚦 SUMO Simulation per Scenario

Inside each folder:

results/routes/pct_XX/

Copy:

map.net.xml
mapB.sumo.cfg
mapD.sumo.cfg

Then run:

🔵 BRD collaborative
sumo-gui mapB.sumo.cfg
🟠 Dijkstra individual
sumo-gui mapD.sumo.cfg
⚠️ For Each Congestion Scenario

For each folder:

pct_10, pct_20 ... pct_100

Copy:

map.net.xml
mapB.sumo.cfg
mapD.sumo.cfg

Then execute SUMO inside each folder independently.

📦 Files that MUST be in GitHub repository
Core code

cos_integrado.py

Example inputs

map.net.xml (example)

od_pairs.txt

mapB.sumo.cfg

mapD.sumo.cfg

Documentation

README.md (this file)

tableIII example output (optional)

📚 Research Context

Repository supporting:

Best Response Dynamics for Collective Route Optimization
IEEE Latin America Transactions (Q1 submission)

📩 Contact

For replication or academic questions:

lourdes.angulo@cinvestav.mx
