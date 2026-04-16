"""Logistics AI Dashboard — Genetic Algorithm Route Optimizer."""

import json
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Logistics AI - Route Optimizer",
    page_icon="\U0001f6f0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stSlider label p {
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #4ade80 !important;
    }

    /* Header area */
    .dashboard-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        border: 1px solid #1e40af33;
    }
    .dashboard-header h1 {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }
    .dashboard-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Section labels */
    .section-label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# City presets (center lat/lon + spread in degrees)
# ---------------------------------------------------------------------------
CITY_PRESETS = {
    "New York, USA": {"lat": 40.7128, "lon": -74.0060, "spread": 0.06},
    "San Francisco, USA": {"lat": 37.7749, "lon": -122.4194, "spread": 0.05},
    "London, UK": {"lat": 51.5074, "lon": -0.1278, "spread": 0.06},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "spread": 0.07},
    "Mumbai, India": {"lat": 19.0760, "lon": 72.8777, "spread": 0.06},
    "Dubai, UAE": {"lat": 25.2048, "lon": 55.2708, "spread": 0.06},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "spread": 0.05},
    "Paris, France": {"lat": 48.8566, "lon": 2.3522, "spread": 0.05},
    "Singapore": {"lat": 1.3521, "lon": 103.8198, "spread": 0.03},
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def generate_city_points(
    n: int, seed: int, center_lat: float, center_lon: float, spread: float
) -> np.ndarray:
    """Return *n* random (lat, lon) points scattered around a city center."""
    rng = np.random.default_rng(seed)
    lats = center_lat + rng.normal(0, spread * 0.5, size=n)
    lons = center_lon + rng.normal(0, spread * 0.5, size=n)
    return np.column_stack([lats, lons])


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


MAP_STYLES = {
    "Google Maps (Light)": {
        "bg": "white-bg",
        "layers": [
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Google Maps",
                "source": [
                    "https://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}"
                ],
            }
        ],
        "marker_text_color": "#1e293b",
        "depot_text_color": "#15803d",
    },
    "Google Satellite": {
        "bg": "white-bg",
        "layers": [
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Google Maps",
                "source": [
                    "https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}"
                ],
            }
        ],
        "marker_text_color": "#ffffff",
        "depot_text_color": "#86efac",
    },
    "Google Terrain": {
        "bg": "white-bg",
        "layers": [
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Google Maps",
                "source": [
                    "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}"
                ],
            }
        ],
        "marker_text_color": "#1e293b",
        "depot_text_color": "#15803d",
    },
    "CartoDB Light": {
        "bg": "carto-positron",
        "layers": [],
        "marker_text_color": "#1e293b",
        "depot_text_color": "#15803d",
    },
    "CartoDB Dark": {
        "bg": "carto-darkmatter",
        "layers": [],
        "marker_text_color": "#e2e8f0",
        "depot_text_color": "#86efac",
    },
    "OpenStreetMap": {
        "bg": "open-street-map",
        "layers": [],
        "marker_text_color": "#1e293b",
        "depot_text_color": "#15803d",
    },
}


def build_map_figure(
    points: np.ndarray,
    route: np.ndarray | None = None,
    center_lat: float = 40.7,
    center_lon: float = -74.0,
    zoom: float = 11.5,
    map_style: str = "Google Maps (Light)",
) -> go.Figure:
    """Build a Plotly Scattermapbox figure with selectable tile layers."""
    style_cfg = MAP_STYLES.get(map_style, MAP_STYLES["Google Maps (Light)"])
    fig = go.Figure()

    lats = points[:, 0]
    lons = points[:, 1]

    # Route lines (drawn first so they appear behind markers)
    if route is not None:
        route_closed = np.append(route, route[0])
        fig.add_trace(
            go.Scattermapbox(
                lat=lats[route_closed],
                lon=lons[route_closed],
                mode="lines",
                line=dict(color="#f43f5e", width=3.5),
                hoverinfo="skip",
                name="Route",
            )
        )

    # Delivery point markers
    labels = [f"Stop {i}" for i in range(len(points))]
    fig.add_trace(
        go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers+text",
            marker=dict(size=14, color="#3b82f6", opacity=0.95),
            text=labels,
            textposition="top center",
            textfont=dict(
                size=10,
                color=style_cfg["marker_text_color"],
                family="Arial Black",
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Lat: %{lat:.4f}<br>"
                "Lon: %{lon:.4f}"
                "<extra></extra>"
            ),
            name="Delivery Points",
        )
    )

    # Depot / start marker
    if route is not None:
        depot = route[0]
        fig.add_trace(
            go.Scattermapbox(
                lat=[lats[depot]],
                lon=[lons[depot]],
                mode="markers+text",
                marker=dict(size=22, color="#22c55e", symbol="circle"),
                text=["DEPOT"],
                textposition="top center",
                textfont=dict(
                    size=11,
                    color=style_cfg["depot_text_color"],
                    family="Arial Black",
                ),
                hovertemplate=(
                    "<b>DEPOT (Start/End)</b><br>"
                    "Lat: %{lat:.4f}<br>"
                    "Lon: %{lon:.4f}"
                    "<extra></extra>"
                ),
                name="Depot",
            )
        )

    mapbox_cfg: dict = {
        "style": style_cfg["bg"],
        "center": dict(lat=center_lat, lon=center_lon),
        "zoom": zoom,
    }
    if style_cfg["layers"]:
        mapbox_cfg["layers"] = style_cfg["layers"]

    fig.update_layout(
        mapbox=mapbox_cfg,
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        showlegend=False,
    )
    return fig


def _sparkline_svg(history: list[float], width: int = 380, height: int = 60) -> str:
    """Return an inline SVG sparkline from history values."""
    if len(history) < 2:
        return ""
    mn, mx = min(history), max(history)
    spread = mx - mn if mx != mn else 1.0
    pad = 4
    pts = []
    for i, v in enumerate(history):
        x = pad + (width - 2 * pad) * i / (len(history) - 1)
        y = pad + (height - 2 * pad) * ((v - mn) / spread)
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(pts)
    area_pts = pts + [f"{width - pad:.1f},{height:.1f}", f"{pad:.1f},{height:.1f}"]
    area_poly = " ".join(area_pts)
    last_x = pts[-1].split(",")[0]
    last_y = pts[-1].split(",")[1]

    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
        xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
      <defs>
        <linearGradient id="areaG" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#818cf8" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="#818cf8" stop-opacity="0.02"/>
        </linearGradient>
        <linearGradient id="lineG" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#a78bfa"/>
          <stop offset="100%" stop-color="#38bdf8"/>
        </linearGradient>
      </defs>
      <polygon points="{area_poly}" fill="url(#areaG)"/>
      <polyline points="{polyline}" fill="none" stroke="url(#lineG)"
                stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="{last_x}" cy="{last_y}" r="4" fill="#38bdf8"
              stroke="#0f172a" stroke-width="1.5"/>
    </svg>"""


def build_progress_html(
    history: list[float],
    current_gen: int,
    total_gens: int,
    initial_distance: float | None = None,
) -> str:
    """Self-contained HTML page for the optimization stats panel."""
    current_dist = history[-1] if history else 0.0
    init_dist = initial_distance if initial_distance else (history[0] if history else 0.0)
    pct = ((init_dist - current_dist) / init_dist * 100) if init_dist > 0 else 0.0
    gen_pct = (current_gen / total_gens * 100) if total_gens > 0 else 0.0

    radius = 52
    circumference = 2 * 3.14159 * radius
    dash = circumference * min(pct, 100) / 100
    gap = circumference - dash

    sparkline = _sparkline_svg(history) if len(history) >= 2 else (
        '<div style="height:40px;display:flex;align-items:center;justify-content:center;'
        'color:#475569;font-size:0.8rem;">Waiting for data...</div>'
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 22px 24px 18px;
    color: #e2e8f0;
  }}
  .row {{ display:flex; align-items:center; gap:20px; margin-bottom:18px; }}
  .stats {{ flex:1; display:flex; flex-direction:column; gap:8px; }}
  .stat-row {{ display:flex; justify-content:space-between; align-items:center; }}
  .label {{ color:#64748b; font-size:0.75rem; font-weight:600;
            text-transform:uppercase; letter-spacing:0.06em; }}
  .val-red {{ color:#f87171; font-size:1rem; font-weight:700; }}
  .val-green {{ color:#4ade80; font-size:1rem; font-weight:700; }}
  .val-blue {{ color:#38bdf8; font-size:1rem; font-weight:700; }}
  .divider {{ height:1px; background:#1e293b; }}
  .section-title {{
    color:#94a3b8; font-size:0.7rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.07em;
  }}
  .bar-bg {{
    height:20px; background:#1e293b; border-radius:10px;
    overflow:hidden; position:relative; margin-top:6px;
  }}
  .bar-fill {{
    position:absolute; top:0; left:0; bottom:0;
    border-radius:10px; transition:width 0.4s ease;
  }}
  .mini-bar {{ height:7px; background:#1e293b; border-radius:4px;
               overflow:hidden; margin-top:5px; }}
  .mini-fill {{ height:100%; border-radius:4px; transition:width 0.3s ease; }}
  .spark-wrap {{ margin-top:5px; }}
</style></head>
<body>

  <!-- Gauge + Stats -->
  <div class="row">
    <svg width="118" height="118" viewBox="0 0 118 118" style="flex-shrink:0;">
      <circle cx="59" cy="59" r="{radius}" fill="none" stroke="#1e293b" stroke-width="9"/>
      <circle cx="59" cy="59" r="{radius}" fill="none"
              stroke="url(#gg)" stroke-width="9"
              stroke-dasharray="{dash:.1f} {gap:.1f}"
              stroke-linecap="round" transform="rotate(-90 59 59)"
              style="transition:stroke-dasharray 0.4s ease;"/>
      <defs>
        <linearGradient id="gg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#818cf8"/>
          <stop offset="100%" stop-color="#38bdf8"/>
        </linearGradient>
      </defs>
      <text x="59" y="53" text-anchor="middle" fill="#f8fafc"
            font-size="21" font-weight="800" font-family="inherit">{pct:.1f}%</text>
      <text x="59" y="70" text-anchor="middle" fill="#64748b"
            font-size="9" font-weight="600" font-family="inherit">OPTIMIZED</text>
    </svg>

    <div class="stats">
      <div class="stat-row">
        <span class="label">Initial</span>
        <span class="val-red">{init_dist:.2f} km</span>
      </div>
      <div class="divider"></div>
      <div class="stat-row">
        <span class="label">Current Best</span>
        <span class="val-green">{current_dist:.2f} km</span>
      </div>
      <div class="divider"></div>
      <div class="stat-row">
        <span class="label">Distance Saved</span>
        <span class="val-blue">{init_dist - current_dist:.2f} km</span>
      </div>
    </div>
  </div>

  <!-- Distance reduction bar -->
  <div style="margin-bottom:16px;">
    <div style="display:flex;justify-content:space-between;">
      <span class="section-title">Distance Reduction</span>
      <span style="color:#e2e8f0;font-size:0.75rem;font-weight:700;">{pct:.1f}%</span>
    </div>
    <div class="bar-bg">
      <div class="bar-fill" style="width:100%;
           background:linear-gradient(90deg,rgba(248,113,113,0.2),rgba(248,113,113,0.06));"></div>
      <div class="bar-fill" style="width:{max(100 - pct, 2):.1f}%;
           background:linear-gradient(90deg,#818cf8,#38bdf8);
           box-shadow:0 0 12px rgba(56,189,248,0.3);"></div>
    </div>
  </div>

  <!-- Sparkline -->
  <div style="margin-bottom:14px;">
    <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
      <span class="section-title">Convergence Curve</span>
      <span style="color:#64748b;font-size:0.7rem;">Gen {current_gen} / {total_gens}</span>
    </div>
    <div class="spark-wrap">{sparkline}</div>
  </div>

  <!-- Evolution progress -->
  <div>
    <div style="display:flex;justify-content:space-between;">
      <span class="section-title">Evolution Progress</span>
      <span style="color:#e2e8f0;font-size:0.75rem;font-weight:700;">{gen_pct:.0f}%</span>
    </div>
    <div class="mini-bar">
      <div class="mini-fill" style="width:{gen_pct:.1f}%;
           background:linear-gradient(90deg,#6366f1,#a78bfa);"></div>
    </div>
  </div>

</body></html>"""


def build_route_json(
    points: np.ndarray, route: np.ndarray, distance: float, city: str
) -> str:
    """Serialize the optimized route to a JSON string."""
    data = {
        "city": city,
        "total_distance_km": round(float(distance), 4),
        "num_stops": len(route),
        "route": [
            {
                "visit_order": idx,
                "stop_id": int(route[idx]),
                "latitude": round(float(points[route[idx], 0]), 6),
                "longitude": round(float(points[route[idx], 1]), 6),
            }
            for idx in range(len(route))
        ],
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Genetic Algorithm engine
# ---------------------------------------------------------------------------
class RouteOptimizer:
    """Travelling-salesman solver using a Genetic Algorithm.

    Uses Haversine distance for geographic accuracy.
    """

    def __init__(
        self,
        points: np.ndarray,
        population_size: int = 100,
        mutation_rate: float = 0.02,
        num_generations: int = 500,
        tournament_size: int = 5,
        elite_count: int = 2,
        rng_seed: int | None = None,
    ) -> None:
        self.points = points
        self.n = len(points)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.tournament_size = min(tournament_size, population_size)
        self.elite_count = min(elite_count, population_size - 1)
        self.rng = np.random.default_rng(rng_seed)

        # Pre-compute pairwise distance matrix (km) for speed
        self._dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = haversine_distance(
                    points[i, 0], points[i, 1], points[j, 0], points[j, 1]
                )
                self._dist_matrix[i, j] = d
                self._dist_matrix[j, i] = d

        self.population: np.ndarray = np.empty((0,), dtype=int)
        self.best_route: np.ndarray = np.empty((0,), dtype=int)
        self.best_distance: float = float("inf")
        self.history: list[float] = []

    def _calculate_distance(self, route: np.ndarray) -> float:
        total = 0.0
        for k in range(len(route) - 1):
            total += self._dist_matrix[route[k], route[k + 1]]
        total += self._dist_matrix[route[-1], route[0]]
        return total

    def _fitness(self, route: np.ndarray) -> float:
        return 1.0 / (self._calculate_distance(route) + 1e-10)

    def initialize_population(self) -> None:
        self.population = np.array(
            [self.rng.permutation(self.n) for _ in range(self.population_size)]
        )
        for ind in self.population:
            d = self._calculate_distance(ind)
            if d < self.best_distance:
                self.best_distance = d
                self.best_route = ind.copy()
        self.history.append(self.best_distance)

    def _tournament_selection(self) -> np.ndarray:
        idxs = self.rng.choice(
            self.population_size, size=self.tournament_size, replace=False
        )
        fits = [self._fitness(self.population[i]) for i in idxs]
        return self.population[idxs[np.argmax(fits)]].copy()

    def _ordered_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        n = self.n
        i, j = sorted(self.rng.choice(n, size=2, replace=False))
        child = np.full(n, -1, dtype=int)
        child[i:j] = p1[i:j]
        segment_set = set(child[i:j].tolist())
        p2_order = [v for v in np.concatenate([p2[j:], p2[:j]]) if v not in segment_set]
        fill_positions = list(range(j, n)) + list(range(0, i))
        for pos, val in zip(fill_positions, p2_order):
            child[pos] = val
        return child

    def _swap_mutation(self, route: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.mutation_rate:
            route = route.copy()
            i, j = self.rng.choice(self.n, size=2, replace=False)
            route[i], route[j] = route[j], route[i]
        return route

    def _evolve_one_generation(self) -> None:
        fitnesses = np.array([self._fitness(ind) for ind in self.population])
        elite_idxs = np.argsort(fitnesses)[-self.elite_count :]
        new_pop: list[np.ndarray] = [self.population[i].copy() for i in elite_idxs]

        while len(new_pop) < self.population_size:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            child = self._ordered_crossover(p1, p2)
            child = self._swap_mutation(child)
            new_pop.append(child)

        self.population = np.array(new_pop[: self.population_size])

        best_idx = int(np.argmax([self._fitness(ind) for ind in self.population]))
        d = self._calculate_distance(self.population[best_idx])
        if d < self.best_distance:
            self.best_distance = d
            self.best_route = self.population[best_idx].copy()
        self.history.append(self.best_distance)

    def run_generation_batch(self, n_gens: int) -> dict:
        for _ in range(n_gens):
            self._evolve_one_generation()
        return {
            "best_route": self.best_route,
            "best_distance": self.best_distance,
            "history": list(self.history),
        }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "points" not in st.session_state:
    default_city = CITY_PRESETS["New York, USA"]
    st.session_state.points = generate_city_points(
        20, 42, default_city["lat"], default_city["lon"], default_city["spread"]
    )
    st.session_state.city_name = "New York, USA"
    st.session_state.best_route = None
    st.session_state.best_distance = None
    st.session_state.history: list[float] = []
    st.session_state.optimized = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center; padding: 12px 0 4px 0;">
        <span style="font-size:2.2rem;">\U0001f6f0</span>
        <h2 style="margin:4px 0 0 0; font-weight:800; letter-spacing:-0.02em;
                    background: linear-gradient(135deg, #60a5fa, #a78bfa);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Route Optimizer
        </h2>
        <p style="font-size:0.78rem; color:#64748b; margin:2px 0 0 0;">
            Powered by Genetic Algorithm
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # -- City & Points -------------------------------------------------------
    st.markdown(
        '<p class="section-label">\U0001f4cd Delivery Zone</p>',
        unsafe_allow_html=True,
    )
    city = st.selectbox("City", list(CITY_PRESETS.keys()), index=0)
    num_locations = st.slider("Delivery stops", 5, 80, 20)
    random_seed = st.number_input("Random seed", 0, 99999, 42)
    btn_generate = st.button(
        "\U0001f504  Generate New Points", use_container_width=True
    )

    st.markdown("---")

    # -- GA params -----------------------------------------------------------
    st.markdown(
        '<p class="section-label">\U0001f9ec Algorithm Settings</p>',
        unsafe_allow_html=True,
    )
    population_size = st.slider("Population size", 20, 500, 100, step=10)
    mutation_rate = st.slider(
        "Mutation rate", 0.001, 0.300, 0.020, step=0.005, format="%.3f"
    )
    num_generations = st.slider("Generations", 50, 2000, 500, step=50)

    st.markdown("---")

    st.markdown(
        '<p class="section-label">\U0001f5fa Map Style</p>',
        unsafe_allow_html=True,
    )
    map_style = st.selectbox(
        "Tile layer", list(MAP_STYLES.keys()), index=0, label_visibility="collapsed"
    )

    with st.expander("Advanced", expanded=False):
        tournament_size = st.slider("Tournament size", 2, 10, 5)
        elite_count = st.slider("Elite count", 1, 10, 2)
        update_every = st.slider("Map refresh interval", 1, 50, 10)

    st.markdown("---")

    btn_optimize = st.button(
        "\U0001f680  Optimize Route", type="primary", use_container_width=True
    )

# ---------------------------------------------------------------------------
# Handle city change or "Generate New Points"
# ---------------------------------------------------------------------------
city_changed = city != st.session_state.get("city_name", "")
if btn_generate or city_changed:
    preset = CITY_PRESETS[city]
    st.session_state.points = generate_city_points(
        num_locations, random_seed, preset["lat"], preset["lon"], preset["spread"]
    )
    st.session_state.city_name = city
    st.session_state.best_route = None
    st.session_state.best_distance = None
    st.session_state.history = []
    st.session_state.optimized = False
    if btn_generate:
        st.rerun()

# ---------------------------------------------------------------------------
# Main area — Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="dashboard-header">
    <h1>\U0001f6f0 Logistics AI Dashboard</h1>
    <p>Real-time delivery route optimization using Genetic Algorithms
       &mdash; watch the AI untangle routes on a live map.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_stops = st.empty()
with m2:
    metric_distance = st.empty()
with m3:
    metric_generation = st.empty()
with m4:
    metric_improvement = st.empty()

# Show default stop count
metric_stops.metric("\U0001f4e6 Stops", len(st.session_state.points))

# ---------------------------------------------------------------------------
# Map + chart area
# ---------------------------------------------------------------------------
col_map, col_right = st.columns([3, 2], gap="medium")

with col_map:
    st.markdown(
        '<p class="section-label">\U0001f5fa Live Route Map</p>',
        unsafe_allow_html=True,
    )
    map_placeholder = st.empty()

with col_right:
    st.markdown(
        '<p class="section-label">\U0001f4e1 Optimization Intelligence</p>',
        unsafe_allow_html=True,
    )
    stats_placeholder = st.empty()

    st.markdown(
        '<p class="section-label" style="margin-top:16px;">\U0001f4cb Route Details</p>',
        unsafe_allow_html=True,
    )
    details_placeholder = st.empty()

status_placeholder = st.empty()
download_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Resolve city centre for map (always from the current selectbox value)
# ---------------------------------------------------------------------------
city_info = CITY_PRESETS[city]

# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------
if btn_optimize and st.session_state.points is not None:
    points = st.session_state.points
    optimizer = RouteOptimizer(
        points=points,
        population_size=population_size,
        mutation_rate=mutation_rate,
        num_generations=num_generations,
        tournament_size=tournament_size,
        elite_count=elite_count,
        rng_seed=int(random_seed),
    )
    optimizer.initialize_population()

    progress = status_placeholder.progress(0, text="Initializing optimizer...")
    gens_done = 0
    initial_distance = optimizer.best_distance

    while gens_done < num_generations:
        batch = min(update_every, num_generations - gens_done)
        result = optimizer.run_generation_batch(batch)
        gens_done += batch

        map_placeholder.plotly_chart(
            build_map_figure(
                points,
                result["best_route"],
                city_info["lat"],
                city_info["lon"],
                map_style=map_style,
            ),
            key=f"map_{gens_done}",
            width="stretch",
        )

        stats_placeholder.html(
            build_progress_html(
                result["history"], gens_done, num_generations, initial_distance
            ),
        )

        metric_stops.metric("\U0001f4e6 Stops", len(points))
        metric_distance.metric(
            "\U0001f4cf Distance",
            f'{result["best_distance"]:.2f} km',
        )
        metric_generation.metric(
            "\U0001f9ec Generation",
            f"{gens_done} / {num_generations}",
        )
        if initial_distance > 0:
            pct = (initial_distance - result["best_distance"]) / initial_distance * 100
            metric_improvement.metric("\U0001f4c8 Improved", f"{pct:.1f} %")

        progress.progress(
            gens_done / num_generations,
            text=f"Evolving \u2014 generation {gens_done} / {num_generations}",
        )

    # Store final results
    st.session_state.best_route = result["best_route"]
    st.session_state.best_distance = result["best_distance"]
    st.session_state.history = result["history"]
    st.session_state.optimized = True
    status_placeholder.success(
        f'Optimization complete!  Best route distance: **{result["best_distance"]:.2f} km**'
    )

    # Show route table
    route = result["best_route"]
    rows = []
    for idx in range(len(route)):
        nxt = route[(idx + 1) % len(route)]
        leg = haversine_distance(
            points[route[idx], 0],
            points[route[idx], 1],
            points[nxt, 0],
            points[nxt, 1],
        )
        rows.append(
            {
                "Order": idx + 1,
                "Stop ID": int(route[idx]),
                "Lat": round(float(points[route[idx], 0]), 4),
                "Lon": round(float(points[route[idx], 1]), 4),
                "Leg (km)": round(leg, 2),
            }
        )
    details_placeholder.dataframe(rows, use_container_width=True, height=230)

# ---------------------------------------------------------------------------
# Show current state (when NOT mid-optimization)
# ---------------------------------------------------------------------------
if not btn_optimize:
    pts = st.session_state.points
    if st.session_state.optimized and st.session_state.best_route is not None:
        map_placeholder.plotly_chart(
            build_map_figure(
                pts,
                st.session_state.best_route,
                city_info["lat"],
                city_info["lon"],
                map_style=map_style,
            ),
            key="map_final",
            width="stretch",
        )
        total_gens = len(st.session_state.history)
        stats_placeholder.html(
            build_progress_html(
                st.session_state.history,
                total_gens,
                total_gens,
                st.session_state.history[0] if st.session_state.history else None,
            ),
        )
        metric_distance.metric(
            "\U0001f4cf Distance",
            f"{st.session_state.best_distance:.2f} km",
        )
        metric_generation.metric(
            "\U0001f9ec Generation",
            f"{total_gens}",
        )
        if len(st.session_state.history) > 1:
            init_d = st.session_state.history[0]
            pct = (init_d - st.session_state.best_distance) / init_d * 100
            metric_improvement.metric("\U0001f4c8 Improved", f"{pct:.1f} %")

        # Route table
        route = st.session_state.best_route
        rows = []
        for idx in range(len(route)):
            nxt = route[(idx + 1) % len(route)]
            leg = haversine_distance(
                pts[route[idx], 0], pts[route[idx], 1],
                pts[nxt, 0], pts[nxt, 1],
            )
            rows.append(
                {
                    "Order": idx + 1,
                    "Stop ID": int(route[idx]),
                    "Lat": round(float(pts[route[idx], 0]), 4),
                    "Lon": round(float(pts[route[idx], 1]), 4),
                    "Leg (km)": round(leg, 2),
                }
            )
        details_placeholder.dataframe(rows, use_container_width=True, height=230)
    else:
        map_placeholder.plotly_chart(
            build_map_figure(pts, center_lat=city_info["lat"], center_lon=city_info["lon"], map_style=map_style),
            key="map_initial",
            width="stretch",
        )

# ---------------------------------------------------------------------------
# Download button
# ---------------------------------------------------------------------------
if st.session_state.optimized and st.session_state.best_route is not None:
    json_str = build_route_json(
        st.session_state.points,
        st.session_state.best_route,
        st.session_state.best_distance,
        st.session_state.get("city_name", "Unknown"),
    )
    download_placeholder.download_button(
        label="\U0001f4e5  Download Optimized Route (JSON)",
        data=json_str,
        file_name="optimized_route.json",
        mime="application/json",
    )
