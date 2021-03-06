# TODO: Decide if you want the objects to store other relevant objects as ID's or references
import functions, pyglet
from graphics import resources

class Junction(object):
    """junction object, every junction has a unique ID which can be autoassigned.
    Stores location of junction and the incident roads."""

    # auto incrementing ID as a class variable
    auto_id = 0

    def __init__(self, x, y, is_exit=False, junction_id=None, *args, **kwargs):
        if junction_id :
            self.junction_id = Junction.auto_id
            Junction.auto_id += 1
        self.location = self.x, self.y = (x, y)
        self.incident_roads = set()
        self.is_exit = is_exit

    def add_road(self, road):
        self.incident_roads.add(road)


class Road(object):
    """Road object, every road has a unique ID which can be autoassigned.
    Stores nodes between which the road lies and th width of the road."""

    # auto incrementing ID as a class variable
    auto_id = 0
    default_width = 30.0

    def __init__(self, start_junction, end_junction, width=None, road_id=None, *args, **kwargs):
        if not road_id:
            self.road_id = Road.auto_id
            Road.auto_id += 1
        else:
            self.road_id = road_id
        self.start_junction, self.end_junction = start_junction, end_junction
        self.width = width if width else Road.default_width
        (startx, starty), (endx, endy) = start_junction.location, end_junction.location
        self.vector = (endx - startx, endy - starty)
        self.length = functions.magnitude(self.vector)
        #self.slope = functions.slope(start_junction.location, end_junction.location)
        lx, ly = functions.weight_add((start_junction.x, start_junction.y),
                                      (end_junction.x, end_junction.y), 0.5, 0.5)
        self.label = pyglet.text.Label(text=str(self.road_id), x=lx, y=ly,batch=kwargs['batch'],
                                      anchor_x='center', anchor_y='center',
                                      font_name='Times New Roman',
                                      font_size=12, color=(255, 0, 0, 255))

class Map(object):
    """Map object, stores all the roads and junctions in a map."""

    def __init__(self, junctions, roads, *args, **kwargs):
        self.roads = set()
        self.junctions = set()
        for road in roads:
            self.add_road(road)
        for junction in junctions:
            self.add_junction(junction)

    def add_road(self, road):
        self.roads.add(road)

    def add_junction(self, junction):
        self.junctions.add(junction)


class Car(pyglet.sprite.Sprite):
    """Car object, stores cars location and velocity"""

    size = resources.car_image.width
    MAX_VEL = 5.0

    def __init__(self, x, y, vx, vy, car_id, road, *args, **kwargs):
        super(Car, self).__init__(x=x, y=y, img=resources.car_image, *args, **kwargs)
        self.velocity = self.vx, self.vy = vx, vy
        self.car_id = car_id
        self.update_road(road)

    def update_road(self, road):
        self.cur_road = road
        if functions.dot( (self.vx, self.vy), road.vector ) > 0:
            self.next_junction = road.end_junction
        else:
            self.next_junction = road.start_junction

    def add_velocity(self, v):
        vx, vy = v
        self.velocity = self.vx, self.vy = min(Car.MAX_VEL, self.vx + vx), min(Car.MAX_VEL, self.vy + vy)

    def __repr__(self):
        return "<Car at ({:.2},{:.2}) with ({:.2},{:.2})>".format(self.x, self.y, self.vx, self.vy)

class ParameterSet(dict):
    def __init__(self,**kw):
        dict.__init__(self,kw)
        self.__dict__ = self