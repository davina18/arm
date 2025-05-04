#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

from robot_model import RobotModel
from tasks.ee_position import EndEffectorPositionTask
from utils import DLS

class TaskPriorityController:
    def __init__(self):
        # Initialise node
        rospy.init_node('task_priority_node')
        self.rate = rospy.Rate(30)

        # Initialise robot model
        self.dof = 4  
        self.q = np.zeros((self.dof, 1))
        self.robot = RobotModel()

        # Publishers and subscribers
        self.arm_vel_pub = rospy.Publisher("/swiftpro/joint_velocity_controller/command", Float64MultiArray, queue_size=1)
        rospy.Subscriber("/swiftpro/joint_states", JointState, self.joint_state_callback)

        # Tasks
        self.tasks = []
        self.initialise_tasks()

        # Marker visualisation
        self.marker_pub = rospy.Publisher("/desired_ee_pos_marker", Marker, queue_size=1)

    def initialise_tasks(self):
        self.tasks.append(EndEffectorPositionTask(value=np.array([0.2, 0.1, 0.0855])))

    def joint_state_callback(self, msg):
        print("Joint state positions from simulator:", msg.position)
        joint_names = {
            "swiftpro/joint1": 0,
            "swiftpro/joint2": 1,
            "swiftpro/joint3": 2,
            "swiftpro/joint4": 3
        }
        for i, name in enumerate(msg.name):
            if name in joint_names:
                self.q[joint_names[name]] = msg.position[i]

    def run(self):
        while not rospy.is_shutdown():
            dq = self.solve()
            q_dot = dq.flatten()

            # Publish arm velocities
            arm_msg = Float64MultiArray(data=q_dot.tolist())
            self.arm_vel_pub.publish(arm_msg)

            # Publish desired EE position marker
            desired_ee_pos = self.tasks[0].get_desired().flatten()
            self.publish_marker(desired_ee_pos)

            self.rate.sleep()

    def solve(self):
        dq = np.zeros((self.dof, 1))
        P = np.eye(self.dof)
        self.robot.update(self.q)

        for task in self.tasks:
            task.update(self.robot)

            if task.is_active():
                J = task.get_jacobian()
                J_bar = J @ P
                err = task.get_error()
                dq_task = DLS(J_bar, damping=0.1) @ (err - J @ dq)
                dq = dq + dq_task
                P = P - np.linalg.pinv(J_bar) @ J_bar

        return dq
    
    def publish_marker(self, position):
        marker = Marker()
        marker.header.frame_id = "swiftpro/manipulator_base_link"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)


if __name__ == '__main__':
    controller = TaskPriorityController()
    controller.run()
