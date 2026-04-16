# SkyRoute-AI

**Real-time delivery route optimization dashboard powered by Genetic Algorithms.**

Watch AI untangle messy delivery routes into optimal paths on a live Google Maps interface — built with Streamlit, Plotly, and NumPy.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3f4f75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

---
<img width="1918" height="898" alt="image" src="https://github.com/user-attachments/assets/3a66d6ae-6010-473b-8249-a28d321cf7bd" />

## Overview

SkyRoute-AI is an interactive logistics dashboard that solves the **Travelling Salesman Problem (TSP)** using a Genetic Algorithm. It generates delivery stops across real-world cities, optimizes the route in real time, and visualizes every generation of the evolution on a live map.

### Key Features

- **Live Map Visualization** — Routes evolve on Google Maps / CartoDB / OpenStreetMap tiles in real time
- **6 Map Styles** — Google Maps (Light), Google Satellite, Google Terrain, CartoDB Light, CartoDB Dark, OpenStreetMap
- **9 City Presets** — New York, San Francisco, London, Tokyo, Mumbai, Dubai, Sydney, Paris, Singapore
- **Optimization Intelligence Panel** — Circular gauge, convergence sparkline, distance reduction bar, and evolution progress
- **Genetic Algorithm Engine** — Tournament Selection, Ordered Crossover (OX1), Swap Mutation, Elitism
- **Haversine Distance** — Geographically accurate route distances in kilometres
- **JSON Export** — Download the optimized route with coordinates for integration with driver apps
- **Fully Configurable** — Tune population size, mutation rate, generations, tournament size, and more

---

## Demo

```
Sidebar Controls          Live Map (Google Maps)        Optimization Intelligence
┌──────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ City: Mumbai     │     │                      │     │   ┌──────┐           │
│ Stops: 30        │     │   Routes untangle    │     │   │58.6% │  Initial  │
│ Population: 100  │     │   in real time on     │     │   │ OPTIM│  130.7 km│
│ Mutation: 0.020  │     │   actual city maps    │     │   └──────┘  Current │
│ Generations: 500 │     │                      │     │              54.2 km │
│                  │     │  ★ DEPOT              │     │ ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔  │
│ [Optimize Route] │     │  ● Stop 1 → Stop 2   │     │ Convergence Curve   │
└──────────────────┘     └──────────────────────┘     │ ╲__________         │
                                                      └──────────────────────┘
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/SkyRoute-AI.git
cd SkyRoute-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

---

## How It Works

### The Genetic Algorithm

The `RouteOptimizer` class implements a classic GA to solve the TSP:

| Step | Method | Description |
|------|--------|-------------|
| **Encoding** | Permutation | Each individual is a permutation of `[0, 1, ..., n-1]` representing visit order |
| **Fitness** | `1 / distance` | Lower total route distance = higher fitness |
| **Selection** | Tournament | Pick `k` random individuals, select the fittest |
| **Crossover** | Ordered (OX1) | Copy a segment from parent 1, fill remainder from parent 2 preserving order |
| **Mutation** | Swap | With probability `p`, swap two random stops |
| **Elitism** | Top-k carry | Best individuals survive unchanged into the next generation |
| **Distance** | Haversine | Great-circle distance between lat/lon coordinates in km |

### Architecture

```
app.py (single file)
├── RouteOptimizer        Pure NumPy GA engine (no Streamlit dependency)
├── build_map_figure()    Plotly Scattermapbox with configurable tile layers
├── build_progress_html() Self-contained HTML/SVG stats panel
├── build_route_json()    JSON export for driver app integration
└── Streamlit UI          Sidebar controls, metrics, live map, download button
```

The `RouteOptimizer` class has zero knowledge of Streamlit — it takes NumPy arrays in, returns dicts out. This means it can be imported and used in any Python project independently.

---

## Configuration

### Sidebar Controls

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| City | 9 presets | New York | Determines geographic center for delivery stops |
| Delivery stops | 5 – 80 | 20 | Number of locations to visit |
| Random seed | 0 – 99999 | 42 | Reproducible point generation |
| Population size | 20 – 500 | 100 | Number of route candidates per generation |
| Mutation rate | 0.001 – 0.300 | 0.020 | Probability of random swap per child |
| Generations | 50 – 2000 | 500 | Total evolution cycles |
| Tournament size | 2 – 10 | 5 | Selection pressure (higher = more greedy) |
| Elite count | 1 – 10 | 2 | Guaranteed survivors per generation |
| Map refresh | 1 – 50 | 10 | Update visuals every N generations |

### Map Styles

| Style | Tiles | Best For |
|-------|-------|----------|
| Google Maps (Light) | Google road map | General use, familiar look |
| Google Satellite | Google aerial imagery | Geographic context |
| Google Terrain | Google topographic | Elevation-aware planning |
| CartoDB Light | Carto Positron | Clean, minimal presentation |
| CartoDB Dark | Carto Dark Matter | Dark-themed dashboards |
| OpenStreetMap | OSM standard | Detailed street-level view |

---

## JSON Export Format

After optimization, click **"Download Optimized Route"** to get a JSON file:

```json
{
  "city": "Mumbai, India",
  "total_distance_km": 54.1923,
  "num_stops": 20,
  "route": [
    {
      "visit_order": 0,
      "stop_id": 7,
      "latitude": 19.0834,
      "longitude": 72.8891
    },
    {
      "visit_order": 1,
      "stop_id": 3,
      "latitude": 19.0612,
      "longitude": 72.8543
    }
  ]
}
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI Framework | Streamlit | Interactive dashboard with sidebar controls |
| Mapping | Plotly Scattermapbox | Real-time route visualization on map tiles |
| Optimization | NumPy + custom GA | Genetic Algorithm for TSP solving |
| Distance | Haversine formula | Geographically accurate route distances |
| Stats Panel | HTML/SVG (inline) | Circular gauge, sparkline, progress bars |
| Export | JSON (stdlib) | Route data for downstream systems |

---

## Project Structure

```
SkyRoute-AI/
├── app.py               # Complete application (GA engine + UI)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Performance

| Scenario | Stops | Population | Generations | Approx. Time |
|----------|-------|------------|-------------|--------------|
| Quick demo | 10 | 50 | 200 | < 2 seconds |
| Standard | 30 | 100 | 500 | ~ 5 seconds |
| Heavy | 50 | 200 | 1000 | ~ 20 seconds |
| Maximum | 80 | 500 | 2000 | ~ 90 seconds |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

<p align="center">
  Built with Streamlit, Plotly, and NumPy<br>
  <strong>SkyRoute-AI</strong> — Smarter routes, faster deliveries.
</p>
