# AS Track Generator

**Overview**

This repository generates random racetrack layouts (cone files) from a bounded Voronoi diagram. It is intended for driving simulations or quick test-track generation. The main outputs are files containing cone coordinates (`cones.mat`, `random_track.csv`, `random_track.yaml`, or `random_track.gpx`).

**Requirements**
- Python 3.11+ and the dependencies listed in `requirements.txt`.

**Quick install**
1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Edit `config.yaml` (see next section), then run the generator:

```bash
python3 main.py
```

**Configuration (`config.yaml`)**
- `mode.parameters`: `random` or `custom`. If `custom`, the values in `track_params` are used; if `random`, parameters are generated automatically.
- `mode.voronoi`: Voronoi selection mode (`RANDOM`, `EXPAND`, `EXTEND`).
- `mode.open_loop`: `true`/`false`. If `true`, the generator produces an "open loop" version of the track (only portions of cones ahead + behind).
- `mode.random_open_loop`: `true`/`false`. If `true`, the `ahead_ratio` and `behind_ratio` are chosen randomly.
- `mode.ahead_ratio`: fraction of cones to keep "ahead" (0.0–1.0).
- `mode.behind_ratio`: fraction of cones to keep "behind" (0.0–1.0). Note: `ahead_ratio + behind_ratio` must not exceed 1.0.
- `mode.missing_cone_ratio`: fraction of cones removed to simulate missing cones (0.0–1.0).

Example: to increase missing cones set `mode.missing_cone_ratio: 0.4`.

**What the script does**
- `main.py` reads `config.yaml`, creates a `TrackGenerator` object and calls `create_track()`.
- The generator builds a bounded Voronoi diagram, selects regions to form the track, interpolates edges, computes left/right cones and saves output files depending on `sim_type`.

**Output**
- `cones.mat`: MATLAB file containing two arrays `cones_left` and `cones_right` (X,Y coordinates).
- Depending on `sim_type`, additional files are generated: CSV, YAML or GPX representing the track.

**MATLAB example**
The repo includes `matlab_example.m` which runs `python3 main.py`, loads `cones.mat` and plots the cones. To use it:

1. Run the generator:

```bash
python3 main.py
```

2. Open `matlab_example.m` in MATLAB/Octave and run it to visualize the cones (or run the MATLAB script directly).


**Note about `open_loop`**
- When `open_loop` is enabled, the function selects two slices of the cone sequence: an "ahead" block (first N cones) and a "behind" block (last M cones). Those slices are concatenated to simulate an open (non-cyclic) track.
- Ensure `ahead_ratio + behind_ratio <= 1.0` — the code will raise an error otherwise.

**Quick edits**
- Change resolution / number of points: edit `track_params.n_points` and `n_regions` in `config.yaml`.
- Increase track width: change `self._track_width` in the `TrackGenerator` constructor (`track_generator.py`) if you need a different default than 3 m.

**Debug / common issues**
- Error: "Unable to find suitable starting position" → try reducing `length_start_area` or increasing `n_regions` / `n_points` in `config.yaml`.
- If generation fails repeatedly, `main.py` randomizes parameters and retries up to 10 times; check printed messages for failure reasons.

