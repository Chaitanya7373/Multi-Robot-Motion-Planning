import matplotlib
import matplotlib.pyplot as plt
from dRRT import DRRT, Node
import plot_helper as mrh

if __name__ == '__main__':
    plot_tree = [True, False] *2
    rrt_params = {
        'iter_max': 100_000,
        'robot_radius': 0.5,
        'step_len': 3.0,
        'move_dist': 0.01, # must be < 0.05 bc that's used in update_robot_position()
        'gamma_FOS': 5.0,
        'epsilon': 0.05,
        'bot_sample_rate': 0.1,
        'waypoint_sample_rate': 0.5,
        'starting_nodes': 500,
        'node_limit': 5000, # for each robot. after this, new nodes only added if robot gets orphaned
    }

top_left = (4, 4)
top_right = (48, 4)
bottom_left = (4, 28)
bottom_right = (48, 28)

r1_start = (top_left[0] + 1, top_left[1] + 1)
r1_goal = (bottom_right[0] - 1, bottom_right[1] - 1)

r2_start = (top_right[0] + 1, top_right[1] + 1)
r2_goal = (bottom_left[0] - 1, bottom_left[1] - 1)

r3_start = (bottom_left[0] + 1, bottom_left[1] + 1)
r3_goal = (top_right[0] - 1, top_right[1] - 1)

r4_start = (bottom_right[0] + 1, bottom_right[1] + 1)
r4_goal = (top_left[0] - 1, top_left[1] - 1)

r1 = DRRT(
        x_origin = r1_start,
        x_goal = r1_goal,
        robot_radius = rrt_params['robot_radius'],
        step_len = rrt_params['step_len'],
        move_dis = rrt_params['move_dist'],
        bot_sample_rate = rrt_params['bot_sample_rate'],
        waypoint_sample_rate = rrt_params['waypoint_sample_rate'],
        start_nodes = rrt_params['starting_nodes'],
        node_limit = rrt_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': True,
            'goal': True,
            'tree': plot_tree[0],
            'path': True,
            'nodes': False,
            'robot_color': 'navy',
            'tree_color': 'navy',
            'path_color': 'navy',
        }
    )

r2 = DRRT(
        x_origin = r2_start,
        x_goal = r2_goal,
        robot_radius = rrt_params['robot_radius'],
        step_len = rrt_params['step_len'],
        move_dis = rrt_params['move_dist'],
        bot_sample_rate = rrt_params['bot_sample_rate'],
        waypoint_sample_rate = rrt_params['waypoint_sample_rate'],
        start_nodes = rrt_params['starting_nodes'],
        node_limit = rrt_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': True,
            'goal': True,
            'tree': plot_tree[1],
            'path': True,
            'nodes': False,
            'robot_color': 'deeppink',
            'tree_color': 'deeppink',
            'path_color': 'deeppink',
        }
    )
r3 = DRRT(
        x_origin = r3_start,
        x_goal = r3_goal,
        robot_radius = rrt_params['robot_radius'],
        step_len = rrt_params['step_len'],
        move_dis = rrt_params['move_dist'],
        bot_sample_rate = rrt_params['bot_sample_rate'],
        waypoint_sample_rate = rrt_params['waypoint_sample_rate'],
        start_nodes = rrt_params['starting_nodes'],
        node_limit = rrt_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': True,
            'goal': True,
            'tree': plot_tree[2],
            'path': True,
            'nodes': False,
            'robot_color': 'firebrick',
            'tree_color': 'firebrick',
            'path_color': 'firebrick',
        }
    )
r4 = DRRT(
        x_origin = r4_start,
        x_goal = r4_goal,
        robot_radius = rrt_params['robot_radius'],
        step_len = rrt_params['step_len'],
        move_dis = rrt_params['move_dist'],
        bot_sample_rate = rrt_params['bot_sample_rate'],
        waypoint_sample_rate = rrt_params['waypoint_sample_rate'],
        start_nodes = rrt_params['starting_nodes'],
        node_limit = rrt_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': True,
            'goal': True,
            'tree': plot_tree[3],
            'path': True,
            'nodes': False,
            'robot_color': 'olivedrab',
            'tree_color': 'olivedrab',
            'path_color': 'olivedrab',
        }
    )
robots = [r1, r2, r3, r4]

for robot in robots:
    robot.set_other_robots([other for other in robots if other != robot])

# plotting stuff
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('DRRT')
ax.set_xlim(r1.map.x_range[0], r1.map.x_range[1]+1)
ax.set_ylim(r1.map.y_range[0], r1.map.y_range[1]+1)

plt.gca().set_aspect('equal', adjustable='box')

plt.pause(0.1)
bg = fig.canvas.copy_from_bbox(ax.bbox)
fig.canvas.blit(ax.bbox)

for i in range(rrt_params['iter_max']):
    # DRRT step for each robot
    for robot in robots:
        robot.step()
    # update plot
    if robots[0].started and i % 30 == 0:
        fig.canvas.restore_region(bg)
        mrh.env_plot(ax, robots[0]) # pass in any robot, they all know the environment
        for robot in robots:
            mrh.single_bot_plot(ax, robot)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

print('\nDRRTcomplete!')


