import numpy as np
import math
import cv2
from PIL import Image, ImageDraw, ImageFont

# class Point:
#     def __init__(self, x, y, z):
#         self.x = x
#         self.y = y
#         self.z = z

#     def point_distance(self, p):
#         distance = math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2 + (self.z - p.z) ** 2)
#         return distance

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def point_distance(self, p):
        distance = math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2 )
        return distance

class Get_angle:
    #计算两个向量之间的角度
    def __init__(self, x1, y1, x2 , y2, x3, y3, x4 ,y4):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        


    def angle(self):
        v1 =[self.x1, self.y1, self.x2, self.y2]
        v2 =[self.x3, self.y3, self.x4, self.y4]
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180/math.pi)
        # print(angle1)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180/math.pi)
        # print(angle2)
        if angle1*angle2 >= 0:
            included_angle = abs(angle1-angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle





class Triangle:
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.distance_p1_p2 = p1.point_distance(p2)
        self.distance_p1_p3 = p1.point_distance(p3)
        self.distance_p2_p3 = p2.point_distance(p3)

    def angle_p1(self):
        cos = (
                      self.distance_p1_p2 ** 2 + self.distance_p1_p3 ** 2 - self.distance_p2_p3 ** 2) / (
                      2 * self.distance_p1_p2 * self.distance_p1_p3)
        angle = round(np.arccos(cos) * 180 / np.pi,2)
        if angle >= 90:
            return 180 - angle
        else:
            return angle

    def angle_p2(self):
        k = self.distance_p1_p2 * self.distance_p2_p3
        if k==0:
            angle = None
        else:
            cos = (
                      self.distance_p1_p2 ** 2 + self.distance_p2_p3 ** 2 - self.distance_p1_p3 ** 2) / (
                      2*self.distance_p1_p2 * self.distance_p2_p3)
            angle = round(np.arccos(cos) * 180 / np.pi,2)
        
        return angle
        # if angle >= 90:
        #     return 180 - angle
        # else:
        #     return angle

    def angle_p3(self):
        cos = (
                      self.distance_p1_p3 ** 2 + self.distance_p2_p3 ** 2 - self.distance_p1_p2 ** 2) / (
                      2 * self.distance_p1_p3 * self.distance_p2_p3)
        angle = round(np.arccos(cos) * 180 / np.pi,2)
        if angle >= 90:
            return 180 - angle
        else:
            return angle
