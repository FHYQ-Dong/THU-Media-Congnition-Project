import time
from pymycobot import MyCobot280
import json


class ArmController():
    def __init__(self, port, baudrate, mapps_file):
        with open(mapps_file, 'r') as f:
            mapps = json.load(f)
            self.grid_map = mapps["grid"]
            self.bin_map = mapps["bin"]
        self.mc = MyCobot280(port, baudrate)
        self.mc.power_on()
        self.mc.focus_all_servos()
        self.init_position(speed=100)
        
    def init_position(self, speed=80):
        self.mc.send_angles([6.85, 58.09, -78.13, -29.44, 1.84, -172.08], speed)
    
    def _dist(self, p1, p2):
        return sum((x1 - x2) ** 2 for x1, x2 in zip(p1, p2)) ** 0.5
        
    def move_to_coords_camera(self, x, y, speed=80):
        """
        x, y, z are in camera coordinate system
        z > 60
        """
        min_dist, min_idx = float('inf'), -1
        for idx, (p, c) in enumerate(self.grid_map):
            dist = self._dist([x, y], p)
            if dist < min_dist:
                min_dist, min_idx = dist, idx
        temp = self.grid_map[min_idx][1]
        angle = [t for t in temp]
        angle[2] = 120
        angle[0] -= 15
        angle[1] -= 15
        self.mc.send_coords(angle, speed)
        time.sleep(2)
        angle[2] = 90
        self.mc.send_coords(angle, speed)
        
    def move_to_angles(self, angles, speed=80):
        """
        angles: [a1, a2, a3, a4, a5, a6]
        """
        return self.mc.send_angles(angles, speed)

    def enable_straw(self):
        self.mc.set_basic_output(2, 1)
        self.mc.set_basic_output(5, 0)
    
    def disable_straw(self):
        self.mc.set_basic_output(5, 1)
        self.mc.set_basic_output(2, 0)
        time.sleep(0.5)
        self.mc.set_basic_output(2, 1)
    
    def pick(self, x, y, speed=80):
        self.move_to_coords_camera(x, y, speed)
        time.sleep(0.5)
        self.enable_straw()
        time.sleep(1)
        self.init_position()
        
    def drop(self, bin_kind, speed=80):
        self.move_to_angles(self.bin_map[bin_kind], speed)
        time.sleep(2)
        self.disable_straw()
        time.sleep(0.5)
        self.init_position()
        