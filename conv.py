import rclpy
import numpy as np
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped 
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry



class FGM_planner(Node):

    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 100  # PREPROCESS_consecutive_SIZE
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees

    def __init__(self, params, robot_scale):
        # super().__init__('fgm_planner')
        self.robot_scale = robot_scale
        self.radians_per_elem = None
        self.STRAIGHTS_SPEED = params['max_speed']
        self.CORNERS_SPEED = params['min_speed']

        self.pub_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.scan_sub = self.create_subscription(LaserScan,'scan',self.scan_callback)
        self.odom_sub = self.create_subscription(Odometry, 'odom',self.odom_callback)

    def preprocess_lidar(self, ranges):

        self.radians_per_elem = (2 * np.pi) / len(ranges)
        proc_ranges = np.array(ranges[180:-180])  # 180도 봄
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)  # 오류 잡이용
        return proc_ranges

    def find_max_gap(self, free_space_ranges):

        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):

        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):

        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2

        return steering_angle

    def plan(self, scan_data, odom_data):
        ranges = scan_data['ranges']
        proc_ranges = self.preprocess_lidar(ranges)
        closest = proc_ranges.argmin()

        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        gap_start, gap_end = self.find_max_gap(proc_ranges)

        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED
        # print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        # print(f"Speed: {speed}")
        return speed, steering_angle



class RosProc(Node):
    def __init__(self):
        self.robot_scale = node.declare_parameter('robot_scale', 0.3302).value

        self.pln_params = {
            'max_speed': node.declare_parameter('max_speed', 0.5).value,
            'min_speed': node.declare_parameter('min_speed', 0.1).value,
            'wpt_path': node.declare_parameter('wpt_path', 'waypoints.csv').value,
            'wpt_delim': node.declare_parameter('wpt_delim', ',').value,
            'wpt_rowskip': node.declare_parameter('wpt_rowskip', 0).value,
            'wpt_xind': node.declare_parameter('wpt_xind', 0).value,
            'wpt_yind': node.declare_parameter('wpt_yind', 1).value,
            'wpt_vind': node.declare_parameter('wpt_vind', 2).value,
            'wpt_thetind': node.declare_parameter('wpt_thetind', 3).value,
        }

        self.planner = FGM_planner(self.pln_params, self.robot_scale)


        self.ackermann = AckermannDriveStamped()
        self.ackermann.drive.speed = 0.0
        self.ackermann.drive.steering_angle = 0.0

        self.scan_data = {'ranges': [0] * 1080}
        self.odom_data = {
            'pose_x': 0.0,
            'pose_y': 0.0,
            'pose_theta': 0.0,
            'linear_vels_x': 0.0,
            'lookahead_distance': 0.82461887897713965,
            'vgain': 0.90338203837889
        }

    def odom_callback(self, odom_msg):
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)

        current_position_theta = np.arctan2(siny_cosp, cosy_cosp)
        current_position_x = odom_msg.pose.pose.position.x
        current_position_y = odom_msg.pose.pose.position.y

        self.odom_data['pose_x'] = current_position_x
        self.odom_data['pose_y'] = current_position_y
        self.odom_data['pose_theta'] = current_position_theta
        self.odom_data['linear_vels_x'] = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        self.scan_data['ranges'] = scan_msg.ranges
        speed, steering_angle = self.planner.plan(self.scan_data, self.odom_data)

        self.ackermann.drive.speed = speed
        self.ackermann.drive.steering_angle = steering_angle
        self.pub.publish(self.ackermann)

# def main(args=None):
#     rclpy.init(args=args)

#     wall_planner = Wall_planner()

#     rclpy.spin(wall_planner)

#     wall_planner.destroy_node()
#     rclpy.shutdown()

    


if __name__ == '__main__':
    rclpy.init()
    node = rclpy.create_node('FGM_convolution')
    app = RosProc()
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass
