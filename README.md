# Last-Mile Delivery Route Optimiser
## Road-Aware TSP with Time Windows — Web App (Vercel) + Desktop (Kivy/Streamlit)

## Live Demo
[**Live Demo →**](https://YOUR-URL.vercel.app)
*(Replace with your Vercel deployment URL after completing the deploy steps below.)*

An end-to-end route optimization system built for small local delivery businesses. Models delivery logistics as a **Traveling Salesperson Problem with Time Windows (TSPTW)** over real road networks extracted from 2D city maps.

---

### Mathematical Model & Units

- **Network Graph**: $G = (V, E)$ where vertex $0$ is the Depot, and $V \setminus \{0\} = \{1, \dots, n-1\}$ are customer delivery locations.
- **Road Distance**: $d(i, j)$ is the true shortest road path distance (in pixels) computed via 8-way A\* search avoiding non-road obstacles.
- **Travel Time**: $t_{\text{travel}}(i, j) = \frac{d(i, j)}{v_{\text{speed}}}$, where $v_{\text{speed}} = 60 \text{ px/min}$ by default.
- **Schedule Simulation**:
  - Vehicle departs depot at unified $t_{\text{start}}$ (default $09:00 = 540\text{ min}$).
  - Arrival at stop $j$: $t_j = t_{\text{depart}, i} + t_{\text{travel}}(i, j)$.
  - If $t_j < e_j$: driver waits until customer window opens ($t_j = e_j$).
  - If $t_j > l_j$: lateness is $\text{late}_j = t_j - l_j$.
  - Departure: $t_{\text{depart}, j} = t_j + s_j$ where $s_j$ is customer service duration (default $5\text{ min}$).
  - Return to depot: checked against depot closing time $t_{\text{close}}$ (default $18:00 = 1080\text{ min}$).
- **Objective Cost Function**:
  $$\text{Score} = \text{Total Distance (px)} + \lambda \cdot v_{\text{speed}} \cdot \sum \text{Lateness (min)}$$
  Both distance and delay penalties are unified into equivalent physical units.

---

### Core Algorithms Implemented

1. **Nearest Neighbour (NN)**: Fast greedy baseline constructing a tour by visiting the closest unvisited stop.
2. **Cheapest Insertion**: Textbook insertion heuristic starting with depot and iteratively inserting the unvisited stop into the position that minimizes incremental routing cost.
3. **2-opt Local Search**: Iterative tour improvement that reverses route segments to eliminate crossings and road detours.
4. **Hill Climbing**: Multi-restart local search with random permutations and 2-opt/swap neighbourhood exploration, with seed control.
5. **Simulated Annealing (SA)**: Metaheuristic with geometric cooling ($T_0=1000, \alpha=0.995$), Metropolis acceptance criterion, reproducible random seeding, and evaluation budgeting.
6. **Exact Branch-and-Bound / Held-Karp**: Computes the proven global optimum for instances up to $N \le 12$ nodes; cleanly refuses larger instances without substituting heuristics.

---

### Key Capabilities

1. **Road-Following A\* Pathfinding**:
   - Isolates road corridors from building blocks via thresholding and morphological filters.
   - Snaps off-road delivery dots to the nearest walkable street pixel.
   - Traces road bends, corners, and intersections on the map rather than drawing straight lines through buildings.

2. **Custom Stop & Address Management**:
   - Add custom delivery stops by entering street addresses and pixel coordinates.
   - Import stops directly from CSV files.
   - Interactive editable time-window table with real-time validation ($e_j \le l_j$).

3. **Multi-Seed Benchmarking & Validation**:
   - Measures Mean, Standard Deviation, Best, Worst, Feasibility %, and Mean Runtime across 10–30 random seeds.
   - Computes exact optimality gap % for every algorithm.
   - Export benchmark comparisons and turn-by-turn route manifests as CSV.

4. **Evaluation-Based Convergence Tracking**:
   - Plots score improvement against total candidate route evaluations on a shared horizontal axis.
   - Displays the exact global optimum as a benchmark reference line.

---

### Quick Setup & Usage

```bash
# Desktop/local dependencies
pip install -r requirements-desktop.txt

# Run the Streamlit web UI locally
streamlit run app.py

# Run the Kivy desktop app
python main.py

# Run the FastAPI web backend locally
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
# Then open public/index.html in a browser pointed at http://localhost:8000

# Run full test suite (49 tests)
python -m pytest test_solver.py test_vision.py test_pathfinding.py -v
```

---

### Deploy to Vercel (Free Hobby Plan)

1. **Install Git:** https://git-scm.com/download/win
2. **Create a GitHub repo** and push this folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit — Vercel deployment"
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git push -u origin main
   ```
3. **Import on Vercel:** https://vercel.com/new → Import Git Repository → pick your repo
   - Framework Preset: **Other**
   - Build Command: *(leave blank)*
   - Output Directory: `public`
   - Install Command: `pip install -r requirements.txt`
4. After deploy, copy the URL and replace `YOUR-URL.vercel.app` at the top of this README.

---

### Project Structure

```
ai_project/
├── app.py                   # Streamlit web application with stop editor & benchmarking
├── main.py                  # Kivy cross-platform mobile/desktop application
├── pathfinding.py           # 8-way A* search engine & road-following polyline renderer
├── vision.py                # OpenCV computer vision detector & labeler
├── solver.py                # 6 TSP-TW algorithms, multi-seed benchmarking & convergence
├── generate_sample_map.py   # Procedural city map generator (roads + buildings)
├── test_pathfinding.py      # Unit tests — A* search & road extraction
├── test_solver.py           # Unit tests — TSP-TW algorithms, seeds, feasibility
├── test_vision.py           # Unit tests — blob detection & coordinates
├── requirements.txt         # Pinned project dependencies
└── README.md                # Comprehensive documentation
```
