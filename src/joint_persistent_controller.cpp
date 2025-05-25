#include "joint_persistent_controller.h"
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace serl_franka_controllers 
{

bool JointPersistentController::init(hardware_interface::RobotHW* robot_hardware, 
                                   ros::NodeHandle& node_handle) 
{
    // Get position joint interface
    position_joint_interface_ = robot_hardware->get<hardware_interface::PositionJointInterface>();
    if (position_joint_interface_ == nullptr) {
        ROS_ERROR("JointPersistentController: Error getting position joint interface from hardware!");
        return false;
    }

    // Initialize joint names for Franka Panda
    joint_names_ = {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", 
                    "panda_joint5", "panda_joint6", "panda_joint7"};
    
    // Get joint handles
    position_joint_handles_.resize(7);
    for (size_t i = 0; i < 7; ++i) {
        try {
            position_joint_handles_[i] = position_joint_interface_->getHandle(joint_names_[i]);
        } catch (const hardware_interface::HardwareInterfaceException& e) {
            ROS_ERROR_STREAM("JointPersistentController: Exception getting joint handles: " << e.what());
            return false;
        }
    }

    // Load parameters from parameter server
    if (!node_handle.getParam("interpolation_time", interpolation_time_)) {
        interpolation_time_ = 0.1;  // Default 100ms interpolation
        ROS_WARN("JointPersistentController: interpolation_time not found, using default: %f", interpolation_time_);
    }

    double max_velocity, max_acceleration;
    if (!node_handle.getParam("max_velocity", max_velocity)) {
        max_velocity_ = 2.0;  // rad/s
        ROS_WARN("JointPersistentController: max_velocity not found, using default: %f", max_velocity_);
    } else {
        max_velocity_ = max_velocity;
    }

    if (!node_handle.getParam("max_acceleration", max_acceleration)) {
        max_acceleration_ = 10.0;  // rad/s^2
        ROS_WARN("JointPersistentController: max_acceleration not found, using default: %f", max_acceleration_);
    } else {
        max_acceleration_ = max_acceleration;
    }

    if (!node_handle.getParam("use_smooth_interpolation", use_smooth_interpolation_)) {
        use_smooth_interpolation_ = true;
        ROS_WARN("JointPersistentController: use_smooth_interpolation not found, using default: true");
    }

    if (!node_handle.getParam("position_tolerance", position_tolerance_)) {
        position_tolerance_ = 0.001;  // 0.001 rad
        ROS_WARN("JointPersistentController: position_tolerance not found, using default: %f", position_tolerance_);
    }

    if (!node_handle.getParam("velocity_filter_gain", velocity_filter_gain_)) {
        velocity_filter_gain_ = 0.1;  // Low-pass filter gain
        ROS_WARN("JointPersistentController: velocity_filter_gain not found, using default: %f", velocity_filter_gain_);
    }

    // Initialize subscribers
    sub_joint_pos_ = node_handle.subscribe("/joint_states_desired", 1, 
                                          &JointPersistentController::jointPositionCallback, this);
    
    // Initialize realtime buffer with default pose
    std::array<double, 7> initial_buffer_pose = reset_pose_;
    target_joint_positions_buffer_.writeFromNonRT(initial_buffer_pose);
    
    // Initialize control variables
    trajectory_active_ = false;
    position_changed_ = false;
    
    // Initialize velocity tracking
    previous_positions_.fill(0.0);
    filtered_velocities_.fill(0.0);
    
    ROS_INFO("JointPersistentController: Initialized successfully");
    ROS_INFO("JointPersistentController: Max velocity: %f rad/s", max_velocity_);
    ROS_INFO("JointPersistentController: Max acceleration: %f rad/s^2", max_acceleration_);
    ROS_INFO("JointPersistentController: Interpolation time: %f s", interpolation_time_);
    
    return true;
}

void JointPersistentController::starting(const ros::Time& time) 
{
    // Get current joint positions
    for (size_t i = 0; i < 7; ++i) {
        initial_pose_[i] = position_joint_handles_[i].getPosition();
        current_target_pose_[i] = initial_pose_[i];
        goal_pose_[i] = initial_pose_[i];
        previous_positions_[i] = initial_pose_[i];
        filtered_velocities_[i] = 0.0;
    }
    
    // Initialize timing
    elapsed_time_ = ros::Duration(0.0);
    trajectory_start_time_ = time;
    trajectory_duration_ = ros::Duration(interpolation_time_);
    
    // Update realtime buffer
    target_joint_positions_buffer_.writeFromNonRT(current_target_pose_);
    
    trajectory_active_ = false;
    position_changed_ = false;
    
    ROS_INFO("JointPersistentController: Started");
}

void JointPersistentController::update(const ros::Time& time, const ros::Duration& period) 
{
    // Read new target positions from realtime buffer
    std::array<double, 7>* target_positions = target_joint_positions_buffer_.readFromRT();
    
    // Check if new target received
    bool new_target = false;
    for (size_t i = 0; i < 7; ++i) {
        if (std::abs((*target_positions)[i] - goal_pose_[i]) > position_tolerance_) {
            new_target = true;
            break;
        }
    }
    
    // Update goal and start new trajectory if needed
    if (new_target) {
        // Store current position as starting point
        for (size_t i = 0; i < 7; ++i) {
            initial_pose_[i] = current_target_pose_[i];
            goal_pose_[i] = (*target_positions)[i];
        }
        
        // Calculate adaptive trajectory duration based on distance and velocity limits
        double max_joint_distance = 0.0;
        for (size_t i = 0; i < 7; ++i) {
            double distance = std::abs(goal_pose_[i] - initial_pose_[i]);
            max_joint_distance = std::max(max_joint_distance, distance);
        }
        
        // Adaptive timing based on maximum distance and velocity constraints
        double required_time = max_joint_distance / max_velocity_;
        trajectory_duration_ = ros::Duration(std::max(required_time, interpolation_time_));
        
        // Reset trajectory timing
        trajectory_start_time_ = time;
        elapsed_time_ = ros::Duration(0.0);
        trajectory_active_ = true;
        
        ROS_DEBUG("JointPersistentController: New target received, trajectory duration: %f s", 
                  trajectory_duration_.toSec());
    }
    
    // Update elapsed time
    elapsed_time_ = time - trajectory_start_time_;
    
    // Generate current target positions
    if (trajectory_active_) {
        double t = elapsed_time_.toSec() / trajectory_duration_.toSec();
        t = std::min(1.0, std::max(0.0, t));  // Clamp to [0, 1]
        
        for (size_t i = 0; i < 7; ++i) {
            if (use_smooth_interpolation_) {
                // Smooth S-curve interpolation for high-dynamic tasks
                current_target_pose_[i] = smoothInterpolation(initial_pose_[i], goal_pose_[i], t);
            } else {
                // Linear interpolation
                current_target_pose_[i] = linearInterpolation(initial_pose_[i], goal_pose_[i], t);
            }
        }
        
        // Check if trajectory completed
        if (t >= 1.0) {
            trajectory_active_ = false;
            ROS_DEBUG("JointPersistentController: Trajectory completed");
        }
    }
    
    // Apply velocity and acceleration limits
    applyDynamicLimits(period);
    
    // Update velocity estimation for monitoring
    updateVelocityEstimation(period);
    
    // Set joint commands
    for (size_t i = 0; i < 7; ++i) {
        position_joint_handles_[i].setCommand(current_target_pose_[i]);
    }
}

void JointPersistentController::stopping(const ros::Time& time) 
{
    ROS_INFO("JointPersistentController: Stopped");
}

void JointPersistentController::jointPositionCallback(const sensor_msgs::JointState::ConstPtr& msg) 
{
    if (msg->position.size() != 7) {
        ROS_WARN("JointPersistentController: Received joint state with %zu positions, expected 7", 
                 msg->position.size());
        return;
    }
    
    std::array<double, 7> new_target;
    for (size_t i = 0; i < 7; ++i) {
        new_target[i] = msg->position[i];
    }
    
    // Write to realtime buffer (thread-safe)
    target_joint_positions_buffer_.writeFromNonRT(new_target);
}

double JointPersistentController::linearInterpolation(double p0, double p1, double t) 
{
    return p0 + t * (p1 - p0);
}

double JointPersistentController::smoothInterpolation(double p0, double p1, double t) 
{
    // S-curve (quintic) interpolation for smooth acceleration/deceleration
    // This provides better performance for high-dynamic tasks
    double t_smooth = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t);
    return p0 + t_smooth * (p1 - p0);
}

void JointPersistentController::applyDynamicLimits(const ros::Duration& period) 
{
    double dt = period.toSec();
    if (dt <= 0.0) return;
    
    for (size_t i = 0; i < 7; ++i) {
        double current_pos = position_joint_handles_[i].getPosition();
        double desired_pos = current_target_pose_[i];
        
        // Calculate desired velocity
        double desired_velocity = (desired_pos - current_pos) / dt;
        
        // Apply velocity limit
        if (std::abs(desired_velocity) > max_velocity_) {
            desired_velocity = std::copysign(max_velocity_, desired_velocity);
        }
        
        // Calculate limited target position
        double limited_target = current_pos + desired_velocity * dt;
        
        // Apply acceleration limit (simplified)
        double velocity_change = desired_velocity - filtered_velocities_[i];
        double max_velocity_change = max_acceleration_ * dt;
        
        if (std::abs(velocity_change) > max_velocity_change) {
            velocity_change = std::copysign(max_velocity_change, velocity_change);
            desired_velocity = filtered_velocities_[i] + velocity_change;
            limited_target = current_pos + desired_velocity * dt;
        }
        
        current_target_pose_[i] = limited_target;
    }
}

void JointPersistentController::updateVelocityEstimation(const ros::Duration& period) 
{
    double dt = period.toSec();
    if (dt <= 0.0) return;
    
    for (size_t i = 0; i < 7; ++i) {
        double current_pos = position_joint_handles_[i].getPosition();
        double raw_velocity = (current_pos - previous_positions_[i]) / dt;
        
        // Apply low-pass filter to velocity estimation
        filtered_velocities_[i] = velocity_filter_gain_ * raw_velocity + 
                                 (1.0 - velocity_filter_gain_) * filtered_velocities_[i];
        
        previous_positions_[i] = current_pos;
    }
}

} // namespace serl_franka_controllers

// Register controller
PLUGINLIB_EXPORT_CLASS(serl_franka_controllers::JointPersistentController, 
                       controller_interface::ControllerBase)