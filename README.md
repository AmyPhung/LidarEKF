# LidarEKF
Experimenting with using an EKF for odometry or SLAM using a lidar

## NEATO Connection
```
roscore
roslaunch neato_node bringup.launch host:=192.168.17.209
```

## Bucket Detection Code:
Derived from https://gitlab.com/concavegit/robot-apf
```
rosrun qea gauntlet.py
```

rviz



Clustering: https://github.com/tysik/obstacle_detector
