---
sidebar_position: 5
---

# ROS 2 Practical Labs

This section contains hands-on labs that reinforce the concepts learned in Module 1. Each lab provides step-by-step instructions for implementing ROS 2 concepts in practical scenarios.

{: .practical-lab}
## Lab 1: Basic Publisher-Subscriber Communication

### Objective
Create a simple publisher and subscriber to understand ROS 2 communication patterns.

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Basic Python knowledge
- Understanding of ROS 2 concepts

### Steps
1. Create a new ROS 2 package:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python my_robot_tutorials
   ```

2. Create a publisher script in `my_robot_tutorials/my_robot_tutorials/talker.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class TalkerNode(Node):
       def __init__(self):
           super().__init__('talker')
           self.publisher = self.create_publisher(String, 'chatter', 10)
           timer_period = 0.5  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = f'Hello World: {self.i}'
           self.publisher.publish(msg)
           self.get_logger().info(f'Publishing: "{msg.data}"')
           self.i += 1

   def main(args=None):
       rclpy.init(args=args)
       talker = TalkerNode()
       rclpy.spin(talker)
       talker.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. Create a subscriber script in `my_robot_tutorials/my_robot_tutorials/listener.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class ListenerNode(Node):
       def __init__(self):
           super().__init__('listener')
           self.subscription = self.create_subscription(
               String,
               'chatter',
               self.listener_callback,
               10)
           self.subscription  # prevent unused variable warning

       def listener_callback(self, msg):
           self.get_logger().info(f'I heard: "{msg.data}"')

   def main(args=None):
       rclpy.init(args=args)
       listener = ListenerNode()
       rclpy.spin(listener)
       listener.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. Update the `setup.py` file to include the new scripts.

5. Build and run the publisher in one terminal:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_tutorials
   source install/setup.bash
   ros2 run my_robot_tutorials talker
   ```

6. In another terminal, run the subscriber:
   ```bash
   cd ~/ros2_ws
   source install/setup.bash
   ros2 run my_robot_tutorials listener
   ```

### Expected Outcome
You should see the publisher sending messages and the subscriber receiving them.

{: .practical-lab}
## Lab 2: Service Implementation

### Objective
Create a service server and client to understand synchronous communication in ROS 2.

### Steps
1. Create a service definition file `AddTwoInts.srv` in a `srv` directory.

2. Implement a service server that adds two integers.

3. Create a service client that requests the addition.

4. Test the service communication.

{: .practical-lab}
## Lab 3: URDF Robot Model Creation

### Objective
Create a URDF model of a simple robot and visualize it in RViz2.

### Steps
1. Create a URDF file describing a simple robot with multiple links and joints.

2. Launch RViz2 to visualize the robot model.

3. Use the robot_state_publisher to publish the robot's joint states.

4. Verify the robot model appears correctly in RViz2.

{: .practical-lab}
## Lab 4: Launch Files

### Objective
Create launch files to coordinate multiple nodes for complex robotic systems.

### Steps
1. Create a launch file that starts multiple nodes simultaneously.

2. Use launch arguments to customize node behavior.

3. Implement conditional launching based on parameters.

4. Test the launch file with different parameter configurations.

## Troubleshooting

Common issues and solutions:
- **Node not connecting**: Check ROS_DOMAIN_ID environment variable
- **Package not found**: Ensure you've sourced the workspace setup file
- **Permission errors**: Check file permissions on scripts
- **Import errors**: Verify dependencies are properly declared in package.xml

## Extensions

For advanced learners:
- Implement a custom message type
- Create a lifecycle node
- Add parameter validation
- Implement a custom launch condition

## Assessment Rubric

Your lab completion will be assessed based on:
- Successful implementation of all required components
- Code quality and adherence to ROS 2 best practices
- Proper documentation and comments
- Correct handling of errors and edge cases