
from transfer import task1
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np


import util
from util.objects import *
from graphics.draw_frame import *
import pyglet

sourcestr = """
    #define GRIDWIDTH {0}
    #define GRIDHEIGHT {1}

    __global__ void task2(float* xsums, float *ysums, float* xsums_out, float *ysums_out)
    {{
        int cellx = blockIdx.x;
        int tid = threadIdx.x;
        if (tid < 9) {{
            int xoff = tid%3 - 1;
            int yoff = tid/3 - 1;
            int x = cellx%GRIDWIDTH;
            int y = cellx/GRIDWIDTH;
            if (x + xoff >= 0 && x+xoff < GRIDWIDTH && y+yoff >= 0 && y+yoff < GRIDHEIGHT) {{
                int flatx = (x+xoff)+GRIDWIDTH*(y+yoff);
                atomicAdd(&xsums_out[cellx], xsums[flatx]);
                atomicAdd(&ysums_out[cellx], ysums[flatx]);
            }}
        }}
    }}
    """


def task2(grid, grid_width, grid_height, scaling):
    mod = SourceModule(sourcestr.format(grid_width, grid_height))
    xsum_ptr, ysum_ptr, res_xsum, res_ysum= task1(grid, grid_width, grid_height)
    print "task 1 xsum", res_xsum
    print "task 1 ysum", res_ysum
    grid_size = grid_width * grid_height
    shp = (grid_size,)
    typ = np.float32
    xsum_out_ptr = cuda.to_device(np.zeros(shp, dtype=typ))
    ysum_out_ptr = cuda.to_device(np.zeros(shp, dtype=typ))
    func = mod.get_function("task2")
    func(xsum_ptr, ysum_ptr, xsum_out_ptr, ysum_out_ptr, grid=(grid_size, 1, 1), block=(1024,1,1))
    pycuda.autoinit.context.synchronize()
    res_xsum = cuda.from_device(xsum_out_ptr, shp, typ)
    res_ysum = cuda.from_device(ysum_out_ptr, shp, typ)
    for y in range(grid_height):
        for x in range(grid_width):
            cur_avgx, cur_avgy = res_xsum[x+y*grid_width], res_ysum[x+y*grid_width]
            for car in grid[(x,y)]:
                # car.vx = -car.vx
                # car.vy = -car.vy
                # car.vx += cur_avgx
                # car.vy += cur_avgy
                car.add_velocity( scale((cur_avgx, cur_avgy), scaling) )


if __name__ == "__main__":
    simple_junctions = [Junction(100, 0, junction_id=0, is_exit=True),
                        Junction(130, 310, junction_id=1),
                        Junction(0, 500, junction_id=2, is_exit=True),
                        Junction(500, 490, junction_id=3),
                        Junction(800, 300, junction_id=4),
                        Junction(300, 390, junction_id=5),
                        Junction(378, 0, junction_id=6, is_exit=True),
                        Junction(278, 500, junction_id=7),
                        Junction(330, 100, junction_id=8),
                        Junction(110, 100, junction_id=9),
                        Junction(0, 150, junction_id=10, is_exit=True),
                        Junction(200, 700, junction_id=11, is_exit=True),
                        Junction(1000, 300, junction_id=12),
                        Junction(1300, 300, junction_id=13, is_exit=True),
                        Junction(800, 100, junction_id=14),
                        Junction(1000, 100, junction_id=15),
                        Junction(1000, 700, junction_id=16, is_exit=True),
                        Junction(500, 700, junction_id=17, is_exit=True),
                        Junction(500, 600, junction_id=18),
                        Junction(700, 600, junction_id=19),
                        Junction(900, 500, junction_id=20),
                        Junction(900, 300, junction_id=21)
                        ]
    # def add_junction(i,j): # adds a node between 2 existing nodes


    # road_conn = [(0, 9), (9, 1), (1, 2), (1, 5), (9, 8), (5, 8), (7, 5), (8, 6), (5, 3), (3, 4)]
    road_conn = [(0, 9), (9, 1), (1, 2), (1, 5), (9, 8), (5, 8), (7, 5), (8, 6),
                 (5, 3), (3, 4), (9, 10), (7, 11), (4, 21), (21, 12), (12, 13),
                 (4, 14), (14, 8), (12, 15), (14, 15), (12, 16), (3, 18),
                 (18, 17), (18, 19), (19, 20), (20, 21)]
    simple_roads = []
    roadsbatch = pyglet.graphics.Batch()
    for start, end in road_conn:
        cur_road = Road(simple_junctions[start], simple_junctions[end], batch=roadsbatch)
        # print cur_road.length
        simple_roads.append(cur_road)
        simple_junctions[start].add_road(cur_road)
        simple_junctions[end].add_road(cur_road)

    curmap = Map(simple_junctions, simple_roads)
    carsbatch = pyglet.graphics.Batch()
    cars = util.load.init_random_cars(curmap, 15, carsbatch, seed=123)
    for i,car in enumerate(cars):
        car.vx = float(i)
        car.vy = float(-i)
    WINWIDTH, WINHEIGHT = 1500, 1000
    cell_width, cell_height = 500, 500
    grid_width, grid_height = WINWIDTH / cell_width, WINHEIGHT / cell_height
    # initialize grid with each cell empty
    grid = {(i, j): [] for i in range(grid_width) for j in range(grid_height)}
    print grid.keys()
    # add cars to grid
    # print grid.keys()
    for i,car in enumerate(cars):
        #gridx, gridy = int(car.x / cell_width), int(car.y / cell_height)
        #grid[(gridx, gridy)].append(car)
        p = i/3
        grid[(p%3, p/3)].append(car)
    print "-"*80 + "\nbefore"
    for k in grid:
        print "in cell", k
        for car in grid[k]:
            print "car id", car.car_id, "velocity:", car.vx, car.vy
    print "-"*80
    task2(grid, grid_width, grid_height, 1.0)
    print "-"*80 + "\nafter"
    for k in grid:
        print "in cell", k
        for car in grid[k]:
            print "car id", car.car_id, "velocity:", car.vx, car.vy