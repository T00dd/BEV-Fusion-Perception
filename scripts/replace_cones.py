import carla

# Replaces every map-placed cone (generic 'static.prop.mesh' actors whose asset
# is a cone) with a physics-enabled cone blueprint that the LiDAR can hit and
# the car can knock over. After the swap it removes ANY leftover cone that was
# not successfully converted, so the scene is left with only the new cones.
#
# Set WIPE_ALL_CONES = True to instead just delete every cone (mesh + blueprint)
# without replacing anything (useful as a clean-slate step).

WIPE_ALL_CONES = True

# Imposta su True per eliminare/nascondere tutti gli oggetti (Actor o Environment Object)
# che contengono la parola 'pole' nel loro nome, tipo o percorso.
DELETE_POLES = True

CONE_MAPPING = {
    'BlueCone':   'static.prop.bluecone',
    'YellowCone': 'static.prop.yellowcone',
    'OrangeCone': 'static.prop.orangecone',
    'RedCone':    'static.prop.redcone',
}

CONE_BLUEPRINT_IDS = set(CONE_MAPPING.values())


def is_mesh_cone(actor):
    if actor.type_id != 'static.prop.mesh':
        return False
    mesh_path = actor.attributes.get('mesh_path', '')
    return any(key in mesh_path for key in CONE_MAPPING)


def matched_blueprint_id(actor):
    mesh_path = actor.attributes.get('mesh_path', '')
    for key, bp_id in CONE_MAPPING.items():
        if key in mesh_path:
            return bp_id
    return None


def collect_all_cones(world):
    cones = []
    for a in world.get_actors():
        tid = a.type_id.lower()
        if tid in CONE_BLUEPRINT_IDS or 'cone' in tid:
            cones.append(a)
        elif is_mesh_cone(a):
            cones.append(a)
    return cones


def wipe_all_cones(client, world):
    cones = collect_all_cones(world)
    if not cones:
        print("No cones found to wipe.")
        return

    # --- AGGIUNTA DEBUG PER LEGGERE I NOMI ---
    print("\n--- NOMI DEI CONI TROVATI (Usa questi in CONE_MAPPING) ---")
    for a in cones:
        # Recupera il nome reale della mesh esportata da UE4
        mesh_path = a.attributes.get('mesh_path', 'Attributo non presente')
        print(f"Actor ID: {a.id} | Type: {a.type_id} | MESH_PATH: {mesh_path}")
    print("-------------------------------------------------------------\n")
    # -----------------------------------------

    client.apply_batch_sync(
        [carla.command.DestroyActor(a.id) for a in cones], True
    )
    print(f"Wiped {len(cones)} cone(s) from the scene.")


def wipe_all_poles(client, world):
    keyword = "pole"   # lowercase; we compare against lowercased strings

    # --- 1. Actor poles (spawned actors and generic static.prop.mesh) -------
    actor_poles = []
    for a in world.get_actors():
        tid = a.type_id.lower()
        mesh_path = a.attributes.get('mesh_path', '').lower()
        if keyword in tid or keyword in mesh_path:
            actor_poles.append(a)

    if actor_poles:
        client.apply_batch_sync(
            [carla.command.DestroyActor(a.id) for a in actor_poles], True
        )
        print(f"Destroyed {len(actor_poles)} pole actor(s).")
    else:
        print("No pole actors found.")

    # --- 2. Environment-object poles (baked map geometry) -------------------
    try:
        env_objects = world.get_environment_objects(carla.CityObjectLabel.Any)
        matches = [obj for obj in env_objects if keyword in obj.name.lower()]
        ids = {obj.id for obj in matches}
        if ids:
            world.enable_environment_objects(ids, False)
            print(f"Disabled {len(ids)} pole environment object(s).")
        else:
            print(f"No environment objects match '{keyword}'.")
    except AttributeError:
        print("This CARLA build does not support get_environment_objects().")


def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        debug = world.debug

        # --- Eliminazione dei pali (Poles) ------------------------------------
        if DELETE_POLES:
            print("Analyzing map for poles...")
            wipe_all_poles(client, world)

        # --- Clean-slate mode: just delete every cone and stop. ---------------
        if WIPE_ALL_CONES:
            wipe_all_cones(client, world)
            return

        # --- 1. Gather the mesh cones we want to replace. --------------------
        print("Analyzing map-placed props for replacement...")
        mesh_cones = [a for a in world.get_actors().filter('static.prop.mesh')
                      if is_mesh_cone(a)]

        spawn_commands = []
        old_actors_to_destroy = []

        for actor in mesh_cones:
            bp_id = matched_blueprint_id(actor)
            blueprint = bp_lib.find(bp_id)
            if blueprint is None:
                continue
            transform = actor.get_transform()
            spawn_commands.append(carla.command.SpawnActor(blueprint, transform))
            old_actors_to_destroy.append(actor)

        if not spawn_commands:
            print("No matching generic cones discovered in this map level.")
            _remove_leftover_cones(client, world, set())
            return

        # --- 2. Batch-spawn the replacements. --------------------------------
        print(f"Found {len(spawn_commands)} matching cones. Replacing...")
        responses = client.apply_batch_sync(spawn_commands, True)

        destroy_commands = []
        new_actor_ids = []

        for response, old_actor in zip(responses, old_actors_to_destroy):
            if response.has_error():
                continue
            new_actor_ids.append(response.actor_id)
            destroy_commands.append(carla.command.DestroyActor(old_actor.id))

        if destroy_commands:
            client.apply_batch_sync(destroy_commands, True)
            print(f"Removed {len(destroy_commands)} converted mesh cones.")

        print("Enabling physics on new cones...")
        for actor_id in new_actor_ids:
            new_cone = world.get_actor(actor_id)
            if new_cone is None:
                continue
            new_cone.set_simulate_physics(True)
            loc = new_cone.get_transform().location
            debug.draw_box(
                box=carla.BoundingBox(loc, carla.Vector3D(0.2, 0.2, 0.4)),
                rotation=new_cone.get_transform().rotation,
                thickness=0.05,
                color=carla.Color(0, 255, 0),
                life_time=60.0,
            )

        _remove_leftover_cones(client, world, set(new_actor_ids))
        print(f"Swap complete! {len(new_actor_ids)} cones upgraded.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Finished script execution.")


def _remove_leftover_cones(client, world, keep_ids):
    leftovers = [a for a in collect_all_cones(world) if a.id not in keep_ids]
    if not leftovers:
        return
    client.apply_batch_sync(
        [carla.command.DestroyActor(a.id) for a in leftovers], True
    )
    print(f"Removed {len(leftovers)} leftover/stale cone(s).")


if __name__ == '__main__':
    main()