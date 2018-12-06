#!/usr/bin/env python

import rospy
import numpy as np

from sensor_msgs.point_cloud2 import read_points

from std_msgs.msg import Float32MultiArray, Int8MultiArray
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

#from gauntlet_lib.potential import attr, repulse
from gauntlet_lib.features import rcirc, multipleRansac
#from gauntlet_lib.motion import turn, forward, stop


class Gauntlet:
    def __init__(self):
        rospy.init_node('gauntlet')

        self.pos = np.float64([0, 0])
        self.hdng = 0
        self.bucketFound = False

        self.lines = np.reshape([], (0, 2, 2))
        self.bucket = np.array([0, 2])
        self.bump = 0
        self.point = PointStamped()
        self.point.header.frame_id = "odom"

        self.subScan = rospy.Subscriber(
            'projected_stable_scan_pc2', PointCloud2, self.scanCallback)
        self.subOdom = rospy.Subscriber('odom', Odometry, self.odomCallback)

        self.pubVel = rospy.Publisher(
            'raw_vel', Float32MultiArray, queue_size=1)
        self.pubBucket = rospy.Publisher('bucket', PointStamped, queue_size=1)

        self.vel = rospy.get_param('vel', .1)
        self.step = rospy.get_param('step', .1)
        self.time = rospy.Time()

    def scanCallback(self, data):
        pts = np.array(list(read_points(data)))[:, :2].T

        lines, other = multipleRansac(
            pts, 512, .005, .08, 5)
        center, other = rcirc(other, .12, 512, .01, 8) #Change this line to tune circle detection

        self.lines = lines
        if center.size != 0:
            self.bucketFound = True
            self.bucket = center
            self.lines = lines
            self.point.header.seq += 1
            self.point.header.stamp = self.time.now()
            self.point.point.x = center[0]
            self.point.point.y = center[1]
        self.pubBucket.publish(self.point)

    def odomCallback(self, data):
        pose = data.pose.pose
        self.pos = np.array([pose.position.x, pose.position.y])
        orientation = pose.orientation
        self.hdng = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])[-1]

 
    def spin(self):
        while not (rospy.is_shutdown()):
            rospy.sleep(.5)
  


if __name__ == '__main__':
    gauntlet = Gauntlet()
    gauntlet.spin()
