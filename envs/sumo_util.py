import numpy as np
import math


def rotationMatrix(rotation):
    return np.array([[-math.sin(rotation), math.cos(rotation)],
                     [math.cos(rotation), math.sin(rotation)]])


class OBB:
    # rotation is radian
    def __init__(self, centerPoint, rotation, halfLength, halfWidth):
        self.centerPoint = centerPoint
        self.halfLength = halfLength
        self.halfWidth = halfWidth
        self.rotation = rotation
        # rotation平行方向
        self.axisX = np.array([-math.sin(rotation), math.cos(rotation)])
        # rotation垂直方向
        self.axisY = np.array([math.cos(rotation), math.sin(rotation)])

    def update_position(self, centerPoint, rotation, halfLength, halfWidth):
        self.centerPoint = centerPoint
        self.halfLength = halfLength
        self.halfWidth = halfWidth
        self.rotation = rotation

        self.axisX = np.array([-math.sin(rotation), math.cos(rotation)])
        self.axisY = np.array([math.cos(rotation), math.sin(rotation)])

    def calcOutline(self):
        rm_r = rotationMatrix(-self.rotation)
        A = np.dot([self.halfLength, self.halfWidth], rm_r) + self.centerPoint
        B = np.dot([self.halfLength, -self.halfWidth], rm_r) + self.centerPoint
        C = np.dot([-self.halfLength, self.halfWidth], rm_r) + self.centerPoint
        D = np.dot([-self.halfLength, -self.halfWidth], rm_r) + self.centerPoint
        return self.centerPoint, A, B, C, D

    def calc_outline(self):
        rm_r = rotationMatrix(-self.rotation)
        A = np.dot([self.halfLength, self.halfWidth], rm_r) + self.centerPoint
        B = np.dot([self.halfLength, -self.halfWidth], rm_r) + self.centerPoint
        C = np.dot([-self.halfLength, self.halfWidth], rm_r) + self.centerPoint
        D = np.dot([-self.halfLength, -self.halfWidth], rm_r) + self.centerPoint
        return A, B, C, D


def isCollision(obb1, obb2):
    axis = [obb1.axisX, obb1.axisY, obb2.axisX, obb2.axisY]
    # centerDistanceVertor = obb1.centerPoint - obb2.centerPoint
    # centerDistanceVertor = centerDistanceVertor / np.linalg.norm(centerDistanceVertor)
    for current_vector in axis:
        cp1 = np.dot(obb1.centerPoint, current_vector)
        cp2 = np.dot(obb2.centerPoint, current_vector)
        max_ = -1e10
        min_ = 1e10
        if cp1 <= cp2:
            o1 = obb1.calc_outline()
            for o in o1:
                max_ = max(max_, np.dot(o, current_vector))
            o2 = obb2.calc_outline()
            for o in o2:
                min_ = min(min_, np.dot(o, current_vector))
        else:
            o2 = obb2.calc_outline()
            for o in o2:
                max_ = max(max_, np.dot(o, current_vector))
            o1 = obb1.calc_outline()
            for o in o1:
                min_ = min(min_, np.dot(o, current_vector))
        if max_ < min_:
            return False
    return True


# move_distance是标量
def clac_center_point(head_point_before_move, rotation, move_distance, length):
    # rotation_voctor = np.array([-math.sin(rotation), math.cos(rotation)])
    rm_r = rotationMatrix(-rotation)
    length = length + move_distance
    head_point = np.dot([move_distance, 0], rm_r) + head_point_before_move
    center_point =head_point- np.dot([length / 2,0], rm_r)

    return center_point

