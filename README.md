# BEV-Fusion-Perception

## Custom Blank Map Setup & OpenDRIVE (.xodr) Integration

This section explains how to properly link a custom empty map (already created in Unreal Engine) with the CARLA Python API, ensuring the simulation can generate the map logic without throwing OpenDRIVE parsing errors.

### 1. Python Environment Compatibility (Client-Server Match)
CARLA is highly sensitive to version mismatches between the simulator (Server) and the Python API (Client). 

You must install the `.whl` file provided in your CARLA build. Please checkout which Python version your `.whl` file.
You might as well rely on a dedicate virtual environment, if so, using `conda`:

   ```bash
   conda create -n carla_env python=<YOUR_SUPPORTED_VERSION>
   conda activate carla_env
   pip install --upgrade pip
   ```

And then,

   ```bash
   cd <CARLA_ROOT>/PythonAPI/carla/dist/
   pip install carla-<CARLA_VERSION>-cp<YOUR_SUPPORTED_VERSION>-cp<YOUR_SUPPORTED_VERSION>-linux_x86_64.whl
   ```
*Note: Make sure your IDE (e.g., VS Code) is set to use the correct matching Python interpreter*

### 2. The Minimal OpenDRIVE (.xodr) File
To use the `world.get_map()` method in Python, CARLA requires a valid `.xodr` (OpenDRIVE) file alongside your visual `.umap` level. If this file is missing or misplaced, you will encounter the `ERROR: unable to parse the OpenDRIVE XML string` fatal error.

Create a minimal `.xodr` file that defines a simple 100-meter straight line (so the simulator has a mathematical reference for the map). 

Create a file named **`YOUR_MAP_NAME.xodr`** (e.g., `void_test_map.xodr`) with the following contents:

```xml
<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="MyBlankMap" version="1.0"
          north="0" south="0" east="0" west="0">
  </header>
  <road name="Road 0" length="100" id="0" junction="-1">
    <link/>
    <planView>
      <geometry s="0" x="0" y="0" hdg="0" length="100">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center>
          <lane id="0" type="none" level="false"/>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
```

### 3. Required Directory Structure
CARLA has strict hardcoded paths for map generation. **Do not place the `.xodr` file in the same directory as your `.umap` file.** The `.xodr` file MUST be placed inside a specific `OpenDrive` subdirectory within the `Maps` folder. 

1. Create the directory if it doesn't exist:
   ```bash
   mkdir -p <CARLA_ROOT>/Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive
   ```
2. Move your newly created XML file there:
   ```bash
   mv void_test_map.xodr <CARLA_ROOT>/Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/
   ```

**Final expected structure:**
```text
Unreal/CarlaUE4/Content/Carla/Maps/
├── void_test_map.umap               <-- Visual 3D level
└── OpenDrive/
    └── void_test_map.xodr           <-- Map logic (MUST match the .umap name)
```

### 4. Verification
With the simulator running your custom map (hit *Play* in Unreal Engine), run the following Python script using your configured `carla_env`:

```python
import carla

# Connect to the simulator
client = carla.Client('localhost', 2000)
client.set_timeout(60.0)

# Load the custom map
world = client.load_world('void_test_map')

# Fetch the map data (this will fail if the .xodr is missing or misplaced)
map_data = world.get_map()
print(f"Success! Map loaded: {map_data.name}")
```
