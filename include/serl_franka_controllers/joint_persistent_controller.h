// Referred to https://github.com/frankaemika/franka_ros/tree/develop/franka_example_controllers
#pragma once

#include <array>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <ros/subscriber.h>
#include <realtime_tools/realtime_buffer.h>

// ROS messages
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <trajectory_msgs/JointTrajectory.h>

namespace serl_franka_controllers
{

class JointPersistentController : public controller_interface::MultiInterfaceController<
    hardware_interface::PositionJointInterface> 
{
public:
    bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
    void starting(const ros::Time&) override;
    void update(const ros::Time&, const ros::Duration& period) override;
    void stopping(const ros::Time&) override;

private:
    // Interpolation function
    double cubicInterpolation(double p0, double p1, double t);
    
    // Hardware interface
    hardware_interface::PositionJointInterface* position_joint_interface_;
    std::vector<hardware_interface::JointHandle> position_joint_handles_;
    
    // Timing
    ros::Duration elapsed_time_;
    ros::Time trajectory_start_time_;
    ros::Duration trajectory_duration_;
    
    // Joint positions
    std::array<double, 7> initial_pose_{};
    std::array<double, 7> current_target_pose_{};
    std::array<double, 7> goal_pose_{};
    std::array<double, 7> reset_pose_{{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    
    // Joint names for Franka Panda
    std::vector<std::string> joint_names_;
    
    // ROS subscribers
    ros::Subscriber sub_joint_pos_;
    ros::Subscriber sub_joint_trajectory_;
    ros::Subscriber sub_joint_array_;
    
    // Realtime buffers for thread-safe communication
    realtime_tools::RealtimeBuffer<std::array<double, 7>> target_joint_positions_buffer_;
    realtime_tools::RealtimeBuffer<trajectory_msgs::JointTrajectory> trajectory_buffer_;
    
    // Callback functions
    void jointPositionCallback(const sensor_msgs::JointState::ConstPtr& msg);
    void jointTrajectoryCallback(const trajectory_msgs::JointTrajectory::ConstPtr& msg);
    void jointArrayCallback(const std_msgs::Float64MultiArray::ConstPtr& msg);
    
    // Control flags
    bool trajectory_active_;
    bool position_changed_;
    
    // Parameters for high-dynamic control
    double interpolation_time_;        // Minimum time to reach target position
    double max_velocity_;             // Maximum joint velocity (rad/s)
    double max_acceleration_;         // Maximum joint acceleration (rad/s^2)
    double position_tolerance_;       // Position tolerance for target detection
    double velocity_filter_gain_;     // Low-pass filter gain for velocity estimation
    bool use_smooth_interpolation_;   // Use S-curve vs linear interpolation
    
    // Dynamic control variables
    std::array<double, 7> previous_positions_;   // For velocity estimation
    std::array<double, 7> filtered_velocities_;  // Filtered velocity estimates
    
    // Additional interpolation functions
    double linearInterpolation(double p0, double p1, double t);
    double smoothInterpolation(double p0, double p1, double t);
    
    // Dynamic limiting functions
    void applyDynamicLimits(const ros::Duration& period);
    void updateVelocityEstimation(const ros::Duration& period);
    
    // Thread safety
    std::mutex target_mutex_;
};

} // namespace serl_franka_controllers