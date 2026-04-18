import carla
from scipy.io import loadmat

BLUEPRINT_CONE_BLUE   = 'static.prop.fssconeblue'   
BLUEPRINT_CONE_YELLOW = 'static.prop.fssconeyellow'

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    bp_blue   = bp_lib.find(BLUEPRINT_CONE_BLUE)
    bp_yellow = bp_lib.find(BLUEPRINT_CONE_YELLOW)

    # Leggi il file .mat
    data        = loadmat('cones.mat')
    cones_left  = data['cones_left']   # blu
    cones_right = data['cones_right']  # gialli

    for x, y in cones_left:
        world.try_spawn_actor(bp_blue, carla.Transform(
            carla.Location(x=float(x), y=float(y), z=0.0)
        ))

    for x, y in cones_right:
        world.try_spawn_actor(bp_yellow, carla.Transform(
            carla.Location(x=float(x), y=float(y), z=0.0)
        ))

    print("Coni spawnati!")

if __name__ == '__main__':
    main()j