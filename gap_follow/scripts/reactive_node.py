#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.lidar_sub = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.lidar_callback,
            10
        )
        
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        self.moving_avg_window = 3
        self.max_dist = 10.0 # meters
        self.bubble_radius = 20 # indices in ranges array
        self.min_obs_dist_threshold = 2.5 # meters

        self.front_view_start_idx = None
        self.front_view_end_idx = None

    
    def range_index_to_angle(self, index, data):
        """ Convert a given index in the LiDAR data.ranges array to an angle in radians
        """
        angle = data.angle_min + index * data.angle_increment
        return angle
    

    def angle_to_index(self, angle, data):
        """ Convert a given angle in radians to an index in the LiDAR data.ranges array
        """
        index = (angle - data.angle_min) / data.angle_increment
        return int(index)
    

    def steering_angle_to_velocity_mapping(self, angle):
        """ Simple mapping between steering angle and velocity
        """
        # Convert steering angle to safe speed
        curr_velocity = 3.0
        if np.abs(angle) < np.radians(10):
            curr_velocity = 3.0
        elif np.abs(angle) < np.radians(20):
            curr_velocity = 2.0
        else:
            curr_velocity = 1.0
        return curr_velocity
    
    
    def lidar_to_steering_angle(self, lidar_angle):
        """ Convert a LiDAR angle to a steering angle
        """
        # As need to map lidar angle from -90 to 90 to steering angle from -45 to 45.
        return lidar_angle / 2.0

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        # Create a copy of the ranges array as a numpy array
        proc_ranges = np.array(ranges)
        # Convert nan values to zero, and higher than max_dist to max_dist
        for i in range(len(proc_ranges)):
            if np.isnan(proc_ranges[i]):
                proc_ranges[i] = 0.0
            elif np.isinf(proc_ranges[i]) or proc_ranges[i] > self.max_dist:
                proc_ranges[i] = self.max_dist

        # Apply a moving average filter using np.convolve()
        kernel = np.ones(self.moving_avg_window) / self.moving_avg_window
        proc_ranges = np.convolve(proc_ranges, kernel, mode='same')

        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """
        Return the start index & end index of the max gap in free_space_ranges
        """
        max_gap = 0
        start_i = 0
        end_i = 0
        gap = 0
        for i in range(len(free_space_ranges)):
            if free_space_ranges[i] > self.min_obs_dist_threshold:
                gap += 1
            else:
                if gap > max_gap:
                    max_gap = gap
                    start_i = i - gap
                    end_i = i
                gap = 0

        # Check the last gap
        if gap > max_gap:
            start_i = len(free_space_ranges) - gap
            end_i = len(free_space_ranges) - 1

        return start_i, end_i


    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        return np.argmax(ranges[start_i:end_i+1]) + start_i


    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges)

        self.front_view_start_idx = self.angle_to_index(-np.pi/2, data)
        self.front_view_end_idx = self.angle_to_index(np.pi/2, data)
        
        # Find closest point to LiDAR - minimum non-zero value
        min_index = self.front_view_start_idx
        min_value = proc_ranges[min_index]
        for i in range(self.front_view_start_idx, self.front_view_end_idx):
            if proc_ranges[i] < min_value and proc_ranges[i] > 0:
                min_value = proc_ranges[i]
                min_index = i

        angle = 0.0
        velocity = self.steering_angle_to_velocity_mapping(angle)
        if min_value != self.max_dist:
            for i in range(min_index-self.bubble_radius, min_index+self.bubble_radius):
                if i >= 0 and i < len(proc_ranges):
                    proc_ranges[i] = 0.0

            #Find max length gap
            start_i, end_i = self.find_max_gap(proc_ranges)

            #Find the best point in the gap
            best_index = self.find_best_point(start_i, end_i, proc_ranges)

            # Choose steering angle and velocity
            angle = self.range_index_to_angle(best_index, data)
            velocity = self.steering_angle_to_velocity_mapping(angle)

        #Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header = data.header
        drive_msg.drive.steering_angle = self.lidar_to_steering_angle(angle)
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    print("Reactive Node Started")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()