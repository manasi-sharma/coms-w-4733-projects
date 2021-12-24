from __future__ import division
from os import link
import sim
import pybullet as p
import random
import numpy as np
import math

MAX_ITERS = 10000
delta_q = 0.1

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)


def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: series of joint angles
    """
    # ========== PART 3 =========
    # TODO: Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    V= set([tuple(q_init)])
    E= set()

    for i in range(MAX_ITERS):
        q_rand= SemiRandomSample(steer_goal_p, q_goal)
        q_nearest= Nearest(V, E, q_rand)
        q_new= Steer(q_nearest, q_rand, delta_q)

        if ObstacleFree(q_nearest, q_new, env):
            V= Union(V, set([tuple(q_new)]))
            E= Union(E, set([ tuple((tuple(q_nearest), tuple(q_new))) ]) )

            visualize_path(q_nearest, q_new, env)

            if (Distance(q_new, q_goal) < delta_q):
                V= Union(V, set([tuple(q_goal)]) )
                E= Union(E, set([ tuple((tuple(q_new), tuple(q_goal))) ]) )
                print("Finding path...")
                path= PathFinder(q_init, q_goal, E, V)
                print("Found path...")
                return path

    return None

def SemiRandomSample(steer_goal_p, q_goal): # I wrote this helper function
    q_rand = np.random.random_sample((6))*2*np.pi-np.pi # range -pi to pi
    arr_random_sample= np.random.choice([0, 1], 1, p=[steer_goal_p, 1-steer_goal_p])
    if(arr_random_sample[0]==0):
        return q_goal
    else:
        return q_rand

def Nearest(V, E, q_rand): # helper function
    nearest_dist= float("inf")
    nearest_vertex= None
    for vertex in V:
        dist= Distance(vertex, q_rand)
        if(dist < nearest_dist):
            nearest_dist= dist
            nearest_vertex= vertex
    return nearest_vertex

def Steer(q_nearest, q_rand, delta_q): # helper function.
    q_nearest= np.array(q_nearest)
    q_rand= np.array(q_rand)
    if(Distance(q_nearest, q_rand) <= delta_q):
        q_new= q_rand
    else:
        dir_vec= ((q_rand-q_nearest)/np.linalg.norm( np.array(q_rand) - np.array(q_nearest) ))*delta_q
        q_new= q_nearest+dir_vec
    return tuple(q_new)

def ObstacleFree(q_nearest, q_new, env): # helper function
    result= env.check_collision(q_new)
    if result:
        return False
    else:
        return True

def Union(set1, set2): # helper function
    return set1.union(set2)

def Distance(q_new, q_goal): # helper function
    return np.linalg.norm(np.array(q_new) - np.array(q_goal))

def PathFinder(q_init, q_goal, E, V): # helper function
    path= list()
    child= tuple(q_goal) #q_init
    #i= 0
    #print("len E: ", len(E))
    while True:
        for edge in E:
            if (child == tuple(q_init)):
                path.append(child)
                break

            if ( edge[1] == child ):
                #print("i: ", i)
                #i+=1
                path.append(child)
                child= edge[0]
        if (child == tuple(q_init)):
            break

    path.reverse()
    return path

def get_grasp_position_angle(object_id):
    position, grasp_angle = np.zeros((3, 1)), 0
    # ========= PART 2============
    # TODO: Get position and orientation (yaw in radians) of the gripper for grasping
    # ==================================
    position, orientation= p.getBasePositionAndOrientation(object_id)
    grasp_angle= p.getEulerFromQuaternion(orientation)[2]
    return position, grasp_angle


if __name__ == "__main__":
    random.seed(1)
    object_shapes = [
        "assets/objects/cube.urdf",
    ]
    env = sim.PyBulletSim(object_shapes = object_shapes)
    num_trials = 3

    # PART 1: Basic robot movement
    # Implement env.move_tool function in sim.py. More details in env.move_tool description
    passed = 0
    for i in range(num_trials):
        # Choose a reachable end-effector position and orientation
        random_position = env._workspace1_bounds[:, 0] + 0.15 + \
            np.random.random_sample((3)) * (env._workspace1_bounds[:, 1] - env._workspace1_bounds[:, 0] - 0.15)
        random_orientation = np.random.random_sample((3)) * np.pi / 4 - np.pi / 8
        random_orientation[1] += np.pi
        random_orientation = p.getQuaternionFromEuler(random_orientation)
        marker = sim.SphereMarker(position=random_position, radius=0.03, orientation=random_orientation)
        # Move tool
        env.move_tool(random_position, random_orientation)
        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
        link_marker = sim.SphereMarker(link_state[0], radius=0.03, orientation=link_state[1], rgba_color=[0, 1, 0, 0.8])
        # Test position
        delta_pos = np.max(np.abs(np.array(link_state[0]) - random_position))
        delta_orn = np.max(np.abs(np.array(link_state[1]) - random_orientation))
        if  delta_pos <= 1e-3 and delta_orn <= 1e-3:
            passed += 1
        env.step_simulation(1000)
        # Return to robot's home configuration
        env.robot_go_home()
        del marker, link_marker
    print(f"[Robot Movement] {passed} / {num_trials} cases passed")

    # PART 2: Grasping
    passed = 0
    env.load_gripper()
    for _ in range(num_trials):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)

        # Test for grasping success (this test is a necessary condition, not sufficient):
        object_z = p.getBasePositionAndOrientation(object_id)[0][2]
        if object_z >= 0.2:
            passed += 1
        env.reset_objects()
    print(f"[Grasping] {passed} / {num_trials} cases passed")

    # PART 3: RRT Implementation
    passed = 0
    for _ in range(num_trials):
        # grasp the object
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            # get a list of robot configuration in small step sizes
            path_conf = rrt(env.robot_home_joint_config,
                            env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.65, env)
            if path_conf is None:
                print(
                    "no collision-free path is found within the time budget. continuing ...")
            else:
                # TODO: Execute the path while visualizing the location of joint 5 (see Figure 2 in homework manual)
                env.set_joint_positions(env.robot_home_joint_config)

                # - For visualization, you can use sim.SphereMarker
                # ===============================================================================\
                markers= list()
                #i= 0
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.1)

                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    markers.append(sim.SphereMarker(link_state[0], radius=0.03, orientation=link_state[1], rgba_color=[1, 0, 0, 0.8]))


                    #i += 1
                # ===============================================================================
                print("Path executed. Dropping the object")

                # TODO: Drop the object
                # - Hint: open gripper, wait, close gripper
                # ===============================================================================
                env.open_gripper()
                env.step_simulation(3)
                env.close_gripper()
                # ===============================================================================

                # TODO: Retrace the path to original location
                # ===============================================================================
                #i= 0
                #markers.reverse()
                for joint_state in list(reversed(path_conf)):
                    env.move_joints(joint_state, speed=0.1)

                while markers:
                    del markers[-1]

                # ===============================================================================
            p.removeAllUserDebugItems()

        env.robot_go_home()

        # Test if the object was actually transferred to the second bin
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and\
            object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and\
            object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()

    print(f"[RRT Object Transfer] {passed} / {num_trials} cases passed")
