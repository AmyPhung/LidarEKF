import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np


def stop(pub):
    pub.publish(Float32MultiArray(data=[0, 0]))


def turn(pub, angle, vel):
    pub.publish(Float32MultiArray(data=np.array([-vel, vel]) * np.sign(angle)))
    rospy.sleep(np.abs(.248 * angle / 2 / vel))
    stop(pub)


def forward(pub, dist, vel):
    pub.publish(Float32MultiArray(data=[vel, vel]))
    rospy.sleep(dist / vel)
    stop(pub)
