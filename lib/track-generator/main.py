import io
import sys
import yaml
import random
import time
from track_generator import TrackGenerator, Mode, SimType

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# --- Load configuration ---
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

mode_choice = config['mode']['parameters'].lower()
voronoi = config['mode']['voronoi'].upper()
open_loop = config['mode']['open_loop']
random_open_loop = config['mode']['random_open_loop']
behind_ratio = config['mode']['behind_ratio']
ahead_ratio = config['mode']['ahead_ratio']
missing_cone_ratio = config['mode']['missing_cone_ratio']

# --- Simulation type mapping ---
sim_type_str = config['simulation']['sim_type'].upper()
if sim_type_str == 'FSDS':
    sim_type = SimType.FSDS
elif sim_type_str == 'FSSIM':
    sim_type = SimType.FSSIM
else:
    sim_type = SimType.GPX

# --- Output options ---
out_cfg = config['output']
plot_track = out_cfg['plot_track']
visualise_voronoi = out_cfg['visualise_voronoi']
create_output_file = out_cfg['create_output_file']
output_location = out_cfg['output_location']

# --- Offsets ---
off = config['offsets']
z_offset = off['z_offset']
lat_offset = off['lat_offset']
lon_offset = off['lon_offset']

# --- Function to randomize parameters ---
def randomize_params():
    n_points = random.randint(40, 100)
    n_regions = random.randint(10, n_points)
    min_bound = random.uniform(0.0, 10.0)
    max_bound = random.uniform(100.0, 200.0)
    mode = random.choice([Mode.EXPAND, Mode.EXTEND, Mode.RANDOM])
    return n_points, n_regions, min_bound, max_bound, mode


# --- If custom mode, use provided params ---
if mode_choice == 'custom':
    params = config['track_params']
    n_points = params['n_points']
    n_regions = params['n_regions']
    min_bound = params['min_bound']
    max_bound = params['max_bound']
    mode = Mode.RANDOM 
else:
    n_points, n_regions, min_bound, max_bound, mode = randomize_params()


# --- Attempt track creation with retry ---
max_attempts = 10
attempt = 0
success = False

while attempt < max_attempts and not success:
    try:
        print(f"\n🔄 Attempt {attempt+1}: Generating track with parameters:")
        print(f"   n_points={n_points}, n_regions={n_regions}, bounds=({min_bound}, {max_bound}), mode={mode.name}")

        track_gen = TrackGenerator(
            n_points=n_points,
            n_regions=n_regions,
            min_bound=min_bound,
            max_bound=max_bound,
            mode=mode,
            open_loop=open_loop,
            missing_cone_ratio=missing_cone_ratio,
            random_open_loop=random_open_loop,
            behind_ratio=behind_ratio,
            ahead_ratio=ahead_ratio,
            plot_track=plot_track,
            visualise_voronoi=visualise_voronoi,
            create_output_file=create_output_file,
            output_location=output_location,
            z_offset=z_offset,
            lat_offset=lat_offset,
            lon_offset=lon_offset,
            sim_type=sim_type
        )

        track_gen.create_track()
        success = True
        print("✅ Track successfully created!")

    except Exception as e:
        print(f"\n⚠️ Unable to create track with the parameters above.")
        print(f"   Reason: {e}")
        print("   Randomizing new parameters and retrying...\n")
        n_points, n_regions, min_bound, max_bound, mode = randomize_params()
        attempt += 1
        time.sleep(0.5)

if not success:
    print("\n❌ Failed to generate a valid track after multiple attempts.")
