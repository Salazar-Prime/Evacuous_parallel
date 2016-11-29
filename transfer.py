import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

from util import load
from util.objects import *
from util import load
from graphics.draw_frame import *
import pyglet
from movement import next_state, global_assignment
from time import time

mod = SourceModule("""
    typedef struct Lists {
        int datalen, __padding;
        float *xptr;
        float *yptr;
    } List;


    __global__ void task1(List *l, float* xsums, float *ysums)
    {
        int cellx = blockIdx.x;
        int caridx = threadIdx.x;
        //if (cellx < 6) {
            if (caridx < l[cellx].datalen) {
                 atomicAdd(xsums+cellx, l[cellx].xptr[caridx]);
                 atomicAdd(ysums+cellx, l[cellx].yptr[caridx]);
                //atomicAdd(xsums, 1);
            }
        //}
    }
    """)


class List():
    int_size = 8
    ptr_size = np.intp(0).nbytes
    mem_size = int_size + 2 * ptr_size

    def __init__(self, arr, list_ptr):
        int_sz, ptr_sz = List.int_size, List.ptr_size
        self.x_ptr = cuda.to_device(arr[0])
        self.y_ptr = cuda.to_device(arr[1])
        self.dtype = arr.dtype
        self.shape = arr[0].shape
        cuda.memcpy_htod(int(list_ptr), np.getbuffer(np.int32(arr.size / 2)))
        cuda.memcpy_htod(int(list_ptr) + int_sz, np.getbuffer(np.intp(int(self.x_ptr))))
        cuda.memcpy_htod(int(list_ptr) + int_sz + ptr_sz, np.getbuffer(np.intp(int(self.y_ptr))))

    def __repr__(self):
        return "{}\n{}".format(cuda.from_device(self.x_ptr, self.shape, self.dtype),
                               cuda.from_device(self.y_ptr, self.shape, self.dtype))

    def __str__(self):
        return repr(self)


def task1_htod(grid, grid_width, grid_height):
    """copies cars in each cell to the device for task1, which aggregates velocities of cars in a cell"""
    num_cars = np.empty((grid_height * grid_width, )).astype(np.int32)
    cars = np.empty((grid_height*grid_width, 512, 2))
    for i in range(grid_height):
        for j in range(grid_width):
            np[i*grid_width + j] = len(grid[(i,j)])

def grid2list(grid, grid_width, grid_height):
    grid_size = grid_width*grid_height
    list_ptr = cuda.mem_alloc(List.mem_size*grid_size)
    for y in range(grid_height):
        for x in range(grid_width):
            cur_idx = y*grid_width + x
            cur_ptr = int(list_ptr) + cur_idx*List.mem_size
            cur_cell = grid[(x, y)]
            if cur_cell:
                vel = np.empty( (2, len(cur_cell)), dtype=np.float32 )
                vel[0] = [car.vx for car in cur_cell]
                vel[1] = [car.vy for car in cur_cell]
            else:
                vel = np.zeros( (2, 1), dtype=np.float32 )
            print vel
            l = List(vel, cur_ptr)
            #print l
    return list_ptr

def task1(grid, grid_width, grid_height):
    list_ptr = grid2list(grid, grid_width, grid_height)
    grid_size = grid_width*grid_height
    shp = (grid_size,)
    typ = np.float32
    xsum_ptr = cuda.to_device( np.zeros(shp, dtype=typ) )
    ysum_ptr = cuda.to_device( np.zeros(shp, dtype=typ) )
    print cuda.from_device(xsum_ptr, shp, typ)
    print cuda.from_device(ysum_ptr, shp, typ)
    func = mod.get_function("task1")
    func(list_ptr, xsum_ptr, ysum_ptr, grid=(grid_size,1,1), block=(32,1,1))
    res_xsum = cuda.from_device(xsum_ptr, shp, typ)
    res_ysum = cuda.from_device(ysum_ptr, shp, typ)
    #xsum_ptr.free(), ysum_ptr.free()
    return res_xsum, res_ysum

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
    cars = load.init_random_cars(curmap, 15, carsbatch, seed=123)

    WINWIDTH, WINHEIGHT = 1500, 1000
    cell_width, cell_height = 500, 500
    grid_width, grid_height = WINWIDTH / cell_width, WINHEIGHT / cell_height
    # initialize grid with each cell empty
    grid = {(i, j): [] for i in range(grid_width) for j in range(grid_height)}

    # add cars to grid
    # print grid.keys()
    for car in cars:
        gridx, gridy = int(car.x / cell_width), int(car.y / cell_height)
        grid[(gridx, gridy)].append(car)
    xsum, ysum = task1(grid, grid_width, grid_height)
    print xsum, ysum
