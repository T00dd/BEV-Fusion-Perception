import carla
from scipy.io import loadmat

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    # -- Verifica disponibilita blueprint --
    bp_blue   = bp_lib.find('static.prop.blue_cone')
    bp_yellow = bp_lib.find('static.prop.yellow_cone')

    print("Blueprint blu   trovato:", bp_blue.id)
    print("Blueprint giallo trovato:", bp_yellow.id)

    # -- Carica le posizioni dal file .mat --
    data        = loadmat('cones.mat')
    cones_left  = data['cones_left']   # coni blu
    cones_right = data['cones_right']  # coni gialli

    # -- Spawn coni blu --
    print("\nSpawn coni blu...")
    for x, y in cones_left:
        try:
            actor = world.spawn_actor(
                bp_blue,
                carla.Transform(carla.Location(x=float(x), y=float(y), z=0.0))
            )
            print(f"  Cono BLU    X:{x:.2f}  Y:{y:.2f}  ID:{actor.id}")
        except Exception as e:
            print(f"  FALLITO BLU X:{x:.2f}  Y:{y:.2f}  -> {e}")

    print("\nSpawn completato.")

    # -- Spawn coni gialli --
    print("\nSpawn coni gialli...")
    for x, y in cones_right:
        try:
            actor = world.spawn_actor(
                bp_yellow,
                carla.Transform(carla.Location(x=float(x), y=float(y), z=0.0))
            )
            print(f"  Cono GIALLO  X:{x:.2f}  Y:{y:.2f}  ID:{actor.id}")
        except Exception as e:
            print(f"  FALLITO GIALLO X:{x:.2f}  Y:{y:.2f}  -> {e}")

    print("\nSpawn completato.")

if __name__ == '__main__':
    main()