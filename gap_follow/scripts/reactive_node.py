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

        self.moving_avg_window = 1
        self.max_dist = 5.0 # meters
        self.car_length = 0.50 # meters
        self.car_width = 0.25 # meters
        self.bubble_radius = 0.20 # meters
        self.min_obs_dist_threshold = 1.5 # meters
        self.disparity_threshold = 0.5 # meters

        self.front_view_start_idx = None
        self.front_view_end_idx = None
        self.angle_increment = None
        self.angle_min = None

        # For debugging
        self.last_printed_angle = -10.0
        self.should_print = False


    def range_index_to_angle(self, index):
        """ Convert a given index in the LiDAR data.ranges array to an angle in radians
        """
        angle = self.angle_min + index * self.angle_increment
        return angle
    

    def angle_to_index(self, angle):
        """ Convert a given angle in radians to an index in the LiDAR data.ranges array
        """
        index = (angle - self.angle_min) / self.angle_increment
        return int(index)
    

    def steering_angle_to_velocity_mapping(self, angle):
        """ Simple mapping between steering angle and velocity
        """
        # Convert steering angle to safe speed
        curr_velocity = 1.2
        if np.abs(angle) < np.radians(10):
            curr_velocity = 1.0
        elif np.abs(angle) < np.radians(20):
            curr_velocity = 0.6
        else:
            curr_velocity = 0.3
        return curr_velocity
    
    
    def lidar_to_steering_angle(self, lidar_angle):
        """ Convert a LiDAR angle to a steering angle
        """
        # As need to map lidar angle from -90 to 90 to steering angle from -45 to 45.
        return lidar_angle #/ 2.0


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


    def bubble_extend(self, ranges, center_idx, extend_value):
        """ Create a bubble around the LiDAR data
        """
        if ranges[center_idx] < self.bubble_radius:
            return

        dtheta = np.arcsin(self.bubble_radius/ranges[center_idx])
        d_index = int(dtheta / self.angle_increment)
        start_idx = center_idx - d_index
        end_idx = center_idx + d_index
        if extend_value:
            for i in range(start_idx, end_idx):
                if i >= 0 and i < len(ranges):
                    ranges[i] = min(ranges[i], ranges[center_idx])

            if self.should_print:
                print("Disparity extended from ", start_idx, " to ", end_idx)
                print(f"Center Index: {center_idx}, Center Value: {ranges[center_idx]}")
        else:
            for i in range(start_idx, end_idx):
                if i >= 0 and i < len(ranges):
                    ranges[i] = 0.0


    def find_deepest_valid_gap(self, free_space_ranges):
        """
        Return the start index & end index of the deepest gap in free_space_ranges
        """
        max_gap_depth = 0
        start_i = self.front_view_start_idx
        end_i = self.front_view_start_idx
        gap_length = 0
        gap_depth = 0
        gap_min_depth = 1000
        for i in range(self.front_view_start_idx, self.front_view_end_idx):
            if free_space_ranges[i] > self.min_obs_dist_threshold:
                gap_length += 1
                gap_depth += free_space_ranges[i]
                gap_min_depth = min(gap_min_depth, free_space_ranges[i])
            else:
                gap_width = gap_length * gap_min_depth * self.angle_increment # r* theta
                if gap_width > 0.8*self.car_width:
                    # Valid gap
                    gap_depth = gap_depth / gap_length # mean gap depth - measure of depth of a gap
                    if self.should_print:
                        print(f"Gap Depth: {gap_depth}, Gap Length: {gap_length}, Gap Min Depth: {gap_min_depth}")

                    # Compare with existing max gap
                    if gap_depth > max_gap_depth:
                        max_gap_depth = gap_depth
                        start_i = i - gap_length
                        end_i = i
                gap_length = 0
                gap_depth = 0
                gap_min_depth = 1000

        # Check the last gap
        if gap_length > 0:
            gap_width = gap_length * gap_min_depth * self.angle_increment
            if gap_width > self.car_width:
                gap_depth = gap_depth / gap_length
                if gap_depth > max_gap_depth:
                    max_gap_depth = gap_depth
                    start_i = self.front_view_end_idx - gap_length
                    end_i = self.front_view_end_idx

        return start_i, end_i


    def find_max_gap(self, free_space_ranges):
        """
        Return the start index & end index of the max gap in free_space_ranges
        """
        max_gap = 0
        start_i = self.front_view_start_idx
        end_i = self.front_view_start_idx
        gap = 0
        for i in range(self.front_view_start_idx, self.front_view_end_idx):
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
            start_i = self.front_view_end_idx - gap
            end_i = self.front_view_end_idx

        return start_i, end_i


    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        # return np.argmax(ranges[start_i:end_i+1]) + start_i
        # Go towards the center of the gap
        return int(start_i*0.55 + 0.45*end_i)
    

    def disparity_extending(self, ranges):
        """
        Find all the disparity in the ranges and create a bubble around it
        to extend the corner reading.
        There is a disparity at point i if |ranges[i] - ranges[i-1]| > disparity_threshold.
        """
        virtual_ranges = ranges.copy()
        for i in range(self.front_view_start_idx+1, self.front_view_end_idx):
            if np.abs(ranges[i] - ranges[i-1]) > self.disparity_threshold:
                if ranges[i] < ranges[i-1]:
                    self.bubble_extend(virtual_ranges, i, True)
                else:
                    self.bubble_extend(virtual_ranges, i-1, True)
        return virtual_ranges


    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges)

        self.angle_increment = data.angle_increment
        self.angle_min = data.angle_min
        self.front_view_start_idx = self.angle_to_index(-np.radians(90))
        self.front_view_end_idx = self.angle_to_index(np.radians(90))

        # Extend disparities
        virtual_ranges = self.disparity_extending(proc_ranges)
        
        # Find closest point to LiDAR - minimum non-zero value
        min_index = self.front_view_start_idx
        min_value = virtual_ranges[self.front_view_start_idx]
        for i in range(self.front_view_start_idx, self.front_view_end_idx):
            if virtual_ranges[i] < min_value:
                min_value = virtual_ranges[i]
                min_index = i

        angle = 0.0
        velocity = self.steering_angle_to_velocity_mapping(angle)
        if min_value != self.max_dist:
            self.bubble_extend(virtual_ranges, min_index, False)

            #Find max length gap
            # start_i, end_i = self.find_max_gap(virtual_ranges)
            # Find deepest valid gap
            start_i, end_i = self.find_deepest_valid_gap(virtual_ranges)
            #Find the best point in the gap
            best_index = self.find_best_point(start_i, end_i, virtual_ranges)

            if (end_i - start_i) < 5:
                # means no gap - just go straight
                angle = 0.0
            else:
                # Choose steering angle and velocity
                angle = self.range_index_to_angle(best_index)

            velocity = self.steering_angle_to_velocity_mapping(angle)

            if (not self.should_print) and np.abs(angle - self.last_printed_angle) > 0.1:
                self.should_print = True
                print(angle - self.last_printed_angle)
                self.last_printed_angle = angle
                print("Front View Start Index: ", self.front_view_start_idx)
                print("Front View End Index: ", self.front_view_end_idx)
                print("Start Index: ", start_i)
                print("End Index: ", end_i)
                print(f"Best Index: {best_index}, Best Value: {virtual_ranges[best_index]}")
                print("Angle: ", angle)
                print("Velocity: ", velocity)
                print("Virtual Ranges: ", virtual_ranges[self.front_view_start_idx:self.front_view_end_idx])
                print("max virtual_ranges: ", max(virtual_ranges))
            else:
                self.should_print = False

        #Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header = data.header
        drive_msg.drive.steering_angle = self.lidar_to_steering_angle(angle)
        drive_msg.drive.speed = velocity #* 0.0
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