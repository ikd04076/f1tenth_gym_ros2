import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped 
from sensor_msgs.msg import LaserScan




class Wall_planner(Node):

    def __init__(self):
        super().__init__('wall_planner')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.subscription = self.create_subscription(LaserScan,'scan',self.callback_1,10)
        self.subscription

    def callback_1(self,scan_msg):
        print(scan_msg.ranges[360])
        msgs = AckermannDriveStamped() 
        msgs.drive.speed= 1.0
        msgs.drive.steering_angle= 0.0 
        self.publisher_.publish(msgs)


def main(args=None):
    rclpy.init(args=args)

    wall_planner = Wall_planner()

    rclpy.spin(wall_planner)

    wall_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
