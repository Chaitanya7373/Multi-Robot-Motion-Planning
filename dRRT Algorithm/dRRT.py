import os
import sys
import time
import math
import queue
from collections import deque
from collections.abc import Sequence
import kdtree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import utils
import map as map

class Node(Sequence):
    def __init__(self,n):
        self.n = n
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.children = set([])

    def __eq__(self, other):
        return id(self) == id(other) or math.hypot(self.x - other.x, self.y -other.y)< 1e-6
    
    def __hash__(self):
        return hash(self.n)
    
    def __getitem__(self,i):
        return self.n[i]
    
    def __len__(self):
        return 2
    
    def set_parent(self,new_parent):
        self.parent = new_parent
        new_parent.children.add(self)

    def distance(self,other):
        return math.hypot(self.x - other.x, self.y - other.y)
    
class DRRT:
    def __init__(self, x_origin, x_goal, robot_radius,step_len, move_dis,
                 bot_sample_rate,waypoint_sample_rate, start_nodes, node_limit = 3000,
                 multi_robot = False, iteration = 10000, plot_params = None):
        self.start = Node(x_origin)
        self.goal = Node(x_goal)
        self.bot = self.start
        self.robot_radius = robot_radius
        self.step_len = step_len
        self.move_dis = move_dis
        self.bot_sample_rate = bot_sample_rate
        self.waypoint_sample_rate = waypoint_sample_rate
        self.start_node = start_nodes
        self.node_limit = node_limit
        self.plot_params = plot_params
        self.kd_tree = kdtree.create([self.goal])
        sys.setrecursionlimit(3000)
        self.tree_node = set([self.goal])
        self.waypoints = []
        self.robot_pos = [self.bot.x,self.bot.y]
        self.robot_speed = 1.0
        self.traveled_distance = 0.0
        self.path = []
        self.other_robots = []
        self.other_robots_obstacles = []
        self.search_radius = 5

        self.utils = utils.Utils()
        # ploting parameters here
        self.started = False
        self.path_to_goal = False
        self.reached_goal = False
        self.regrowing = False

        self.obs_robot = []
        self.map = map.Map()

        self.x_range = self.map.x_range
        self.y_range = self.map.y_range
        self.obs_circle = self.map.obs_circle
        self.obs_rectangle = self.map.obs_rectangle
       #self.obs_boundary = self.map.obs_boundary
        self.obs_robot = []

    def step(self):
        if self.reached_goal:
            return
        
        if not self.started and len(self.tree_node) > self.start_node:
            self.started = True

        if self.started and self.path_to_goal:
            self.bot,self.robot_pos = self.utils.update_robot_position(
                self.robot_pos,self.bot,self.robot_speed,self.move_dis
            )
            self.traveled_distance += self.robot_speed*self.move_dis

        if self.bot == self.goal:
            self.reached_goal = True
        
        self.update_robot_obstacles(delta=0.5)
        if len(self.tree_node) >= self.node_limit and self.path_to_goal:
            return
        if self.regrowing:
            v = self.random_node_regrow()
        else:
            v = self.random_node()

        v_nearest = self.nearest(v)
        v = self.saturate(v_nearest,v)

        if v and not self.utils.is_collision(v_nearest,v):
            self.extend(v,v_nearest)

    def set_other_robots(self,other_robots):
        self.other_robots = other_robots
        self.other_robots_obstacles = []
        for robot in other_robots:
            self.other_robots_obstacles.append([
                robot.robot_pos[0],
                robot.robot_pos[1],
                robot.robot_radius
            ])

    def update_robot_obstacles(self,delta):
        delta_i = []
        for i, other in enumerate(self.other_robots):
            if math.hypot(other.robot_pos[0]-self.other_robots_obstacles[i][0],
                          other.robot_pos[1]-self.other_robots_obstacles[i][1])>delta:
                delta_i.append(i)

        #remove obstacles from old positions
        for i in delta_i:
            self.remove_obstacle(self.other_robots_obstacles[i],'robot')

        for i in delta_i:
            self.other_robots_obstacles[i]=[
                self.other_robots[i].robot_pos[0],
                self.other_robots[i].robot_pos[1],
                self.other_robots[i].robot_radius
            ]
            self.add_new_obstacles(self.other_robots_obstacles[i], robot = True)

    def remove_obstacle(self, obs, shape):
        if self.obs_robot:
            self.obs_robot.pop()

        # remove obstacle from list if applicable
        if shape == 'circle':
            self.obs_circle.remove(obs)
            self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking
        elif shape == 'rectangle':
            self.obs_rectangle.remove(obs)
            self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking

    def add_node(self, node_new):
        if node_new not in self.tree_node:
            self.tree_node.add(node_new)
            self.kd_tree.add(node_new)
        # if new node is at start, then path to goal is found
        if node_new == self.bot:
            self.bot = node_new
            self.path_to_goal = True
            self.regrowing = False
            self.update_path(self.bot) # update path to goal for plotting

    def update_path(self, node):
        self.path = []
        while node.parent:
            self.path.append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
            node = node.parent

    def near(self, v):
        return self.kd_tree.search_nn_dist((v.x, v.y), self.search_radius)
    
    def random_node(self):
        #uniform random sampling
        delta = self.utils.delta

        if not self.path_to_goal and np.random.random()<self.bot_sample_rate:
            return Node(self.bot.n)
        
        return Node((np.random.uniform(self.x_range[0]+delta, self.x_range[1]-delta),
                     np.random.uniform(self.y_range[0]+delta, self.y_range[1]-delta)))
    

    def random_node_regrow(self):
        delta = self.utils.delta

        if not self.path_to_goal and np.random.random()<self.bot_sample_rate:
            return Node(self.bot.n)
        
        return Node((np.random.uniform(self.x_range[0]+delta, self.x_range[1]-delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1]-delta)))
    
    def nearest(self, v):
        return self.kd_tree.search_nn((v.x, v.y))[0].data
    
    def extend(self, v, v_nearest):
        v.set_parent(v_nearest)
        self.add_node(v)
    
    def find_nodes_in_range(self, pos, r):
        return self.kd_tree.search_nn_dist((pos[0], pos[1]), r)
    
    def find_obstacle(self,a,b):
        for(x,y,r) in self.obs_circle:
            if math.hypot(a-x,b-y)<= r:
                return ([x,y,r], 'circle')
        for(x,y,w,h) in self.obs_rectangle:
            if 0 <= a - (x) <= w+2 and 0<= b-(y) <= h:
                return ([x,y,w,h],'rectangle')

    def add_new_obstacles(self,obs,robot):
        x,y,r = obs
        self.obs_robot.append(obs)

        #re-check collision with new obstacles
        nearby_node = self.find_nodes_in_range((x,y),r+self.step_len+self.utils.delta)
        set = [i for i in nearby_node if i.parent and self.utils.is_intersect_circle(i.n, i.parent.n,[x,y],r)]
        if not set:
            return
        
        #remove nodes as parents' children
        for i in set:
            i.parent.children.remove(i)

        #remove children from tree recursively
        q = deque(set)
        while q:
            node = q.pop()
            if node == self.bot:
                self.path_to_goal = False
                self.regrowing = True
            for child in node.children:
                q.appendleft(child)

            try:
                self.tree_node.remove(node)
                self.kd_tree.remove(node)
            except KeyError:
                pass

        #update waypoints
        self.waypoints=[]
        for edge in self.path:
            pos = tuple(edge[0,:])
            node = Node(pos)
            if not node in self.tree_node:
                self.waypoints.append(pos)
            else:
                break

    def saturate(self, v_nearest, v):
        dist, theta = self.get_distance_and_angle(v_nearest, v)
        dist = min(self.step_len, dist)
        node_new = Node((v_nearest.x + dist * math.cos(theta),
                         v_nearest.y + dist * math.sin(theta)))
        return node_new
    
    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    @staticmethod
    def get_distance(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy)
