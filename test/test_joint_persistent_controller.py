#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import Twist

class FrankaTestTrajectoryPublisher:
    def __init__(self):
        rospy.init_node('franka_test_publisher', anonymous=True)
        
        # ROS Publishers and Subscribers
        self.target_pub = rospy.Publisher('/joint_states_desired', JointState, queue_size=10)
        self.current_sub = rospy.Subscriber('/joint_states', JointState, self.current_state_callback)
        
        # Franka Panda joint names
        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        
        # Safe home position (Franka ready pose)
        self.home_position = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
        
        # Current joint states
        self.current_positions = np.copy(self.home_position)
        self.current_received = False
        
        # Trajectory parameters
        self.publish_rate = 50.0  # Hz
        self.trajectory_type = 'sine_wave'  # 'sine_wave', 'step', 'circular', 'figure_eight'
        
        # Test parameters (safe ranges)
        self.amplitude = 0.3      # Maximum 0.3 rad deviation from home
        self.frequency = 0.1      # 0.1 Hz base frequency
        self.test_duration = 30.0 # 30 seconds test
        
        # Tracking error monitoring
        self.max_history = int(self.publish_rate * 10)  # 10 seconds of data
        self.target_history = deque(maxlen=self.max_history)
        self.actual_history = deque(maxlen=self.max_history)
        self.error_history = deque(maxlen=self.max_history)
        self.time_history = deque(maxlen=self.max_history)
        
        # Statistics
        self.max_errors = np.zeros(7)
        self.rms_errors = np.zeros(7)
        self.start_time = None
        
        # Thread for plotting
        self.plot_thread = None
        self.plotting_enabled = True
        
        rospy.loginfo("Franka Test Publisher initialized")
        rospy.loginfo(f"Publishing at {self.publish_rate} Hz")
        rospy.loginfo(f"Test trajectory: {self.trajectory_type}")
        rospy.loginfo(f"Test duration: {self.test_duration} seconds")
        
    def current_state_callback(self, msg):
        """Callback for current joint states"""
        if len(msg.position) >= 7:
            self.current_positions = np.array(msg.position[:7])
            self.current_received = True
    
    def generate_safe_trajectory(self, t):
        """Generate safe test trajectories"""
        target = np.copy(self.home_position)
        
        if self.trajectory_type == 'sine_wave':
            # Sine wave on alternating joints (safe)
            for i in range(7):
                if i % 2 == 0:  # Joints 1, 3, 5, 7
                    phase = i * np.pi / 4
                    target[i] += self.amplitude * np.sin(2 * np.pi * self.frequency * t + phase)
                    
        elif self.trajectory_type == 'step':
            # Step trajectory with smooth transitions
            step_duration = 3.0  # 3 seconds per step
            step_num = int(t / step_duration) % 4
            transition_time = 0.5  # 0.5 second smooth transition
            
            # Target positions for each step
            steps = [
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                np.array([0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.2, -0.2, 0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0])
            ]
            
            current_step = steps[step_num]
            next_step = steps[(step_num + 1) % 4]
            
            # Smooth transition
            t_in_step = t % step_duration
            if t_in_step < transition_time:
                alpha = t_in_step / transition_time
                # S-curve interpolation
                alpha = alpha * alpha * (3 - 2 * alpha)
                step_offset = (1 - alpha) * steps[(step_num - 1) % 4] + alpha * current_step
            else:
                step_offset = current_step
                
            target += step_offset
            
        elif self.trajectory_type == 'circular':
            # Circular motion in joint space (joints 1 and 2)
            angle = 2 * np.pi * self.frequency * t
            target[0] += self.amplitude * np.cos(angle)
            target[1] += self.amplitude * np.sin(angle) * 0.5  # Reduced amplitude for joint 2
            
        elif self.trajectory_type == 'figure_eight':
            # Figure-eight trajectory (joints 1, 2, 3)
            angle = 2 * np.pi * self.frequency * t
            target[0] += self.amplitude * np.sin(angle)
            target[1] += self.amplitude * np.sin(2 * angle) * 0.5
            target[2] += self.amplitude * np.cos(angle) * 0.3
            
        # Safety limits check
        joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        target = np.clip(target, joint_limits_lower, joint_limits_upper)
        
        return target
    
    def update_tracking_error(self, target_pos, current_pos, timestamp):
        """Update tracking error statistics"""
        if not self.current_received:
            return
            
        error = target_pos - current_pos
        
        # Store history
        self.target_history.append(target_pos.copy())
        self.actual_history.append(current_pos.copy())
        self.error_history.append(error.copy())
        self.time_history.append(timestamp)
        
        # Update statistics
        abs_error = np.abs(error)
        self.max_errors = np.maximum(self.max_errors, abs_error)
        
        # Calculate RMS error over recent history
        if len(self.error_history) > 10:
            recent_errors = np.array(list(self.error_history)[-100:])  # Last 2 seconds
            self.rms_errors = np.sqrt(np.mean(recent_errors**2, axis=0))
    
    def print_status(self, t, target_pos, error):
        """Print current status"""
        print(f"\n--- Time: {t:.2f}s ---")
        print(f"Target:  [{', '.join([f'{x:6.3f}' for x in target_pos])}]")
        print(f"Current: [{', '.join([f'{x:6.3f}' for x in self.current_positions])}]")
        print(f"Error:   [{', '.join([f'{x:6.3f}' for x in error])}]")
        print(f"Max Err: [{', '.join([f'{x:6.3f}' for x in self.max_errors])}]")
        print(f"RMS Err: [{', '.join([f'{x:6.3f}' for x in self.rms_errors])}]")
        print(f"Max Total Error: {np.max(np.abs(error)):.4f} rad ({np.degrees(np.max(np.abs(error))):.2f} deg)")
    
    def plot_results(self, axes):
        """Plot tracking results in real-time"""
        if not self.plotting_enabled or len(self.error_history) < 10:
            return
            
        try:

            times = np.array(list(self.time_history)[-500:])  # Last 10 seconds
            if len(times) == 0:
                return
                
            times = times - times[0]  # Relative time
            
            # Plot 1: Joint positions
            targets = np.array(list(self.target_history)[-500:])
            actuals = np.array(list(self.actual_history)[-500:])
            
            axes[0].clear()
            for i in range(7):
                axes[0].plot(times, targets[:, i], f'C{i}--', alpha=0.7, label=f'Target J{i+1}')
                axes[0].plot(times, actuals[:, i], f'C{i}-', label=f'Actual J{i+1}')
            axes[0].set_ylabel('Position (rad)')
            axes[0].set_title('Joint Positions')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True)
            
            # Plot 2: Tracking errors
            errors = np.array(list(self.error_history)[-500:])
            axes[1].clear()
            for i in range(7):
                axes[1].plot(times, errors[:, i], f'C{i}-', label=f'Error J{i+1}')
            axes[1].set_ylabel('Error (rad)')
            axes[1].set_title('Tracking Errors')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True)
            
            # Plot 3: Error magnitude
            error_mag = np.linalg.norm(errors, axis=1)
            axes[2].clear()
            axes[2].plot(times, error_mag, 'r-', linewidth=2, label='Total Error')
            axes[2].plot(times, np.full_like(times, 0.01), 'g--', alpha=0.7, label='1cm threshold')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Error Magnitude (rad)')
            axes[2].set_title('Total Tracking Error')
            axes[2].legend()
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.pause(0.01)
            
        except Exception as e:
            rospy.logwarn(f"Plotting error: {e}")
    
    def run_test(self):
        """Run the test trajectory"""
        rospy.loginfo("Starting test trajectory...")
        rospy.loginfo("Waiting for current joint states...")
        
        # Wait for current joint states
        while not self.current_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        rospy.loginfo("Current joint states received. Starting trajectory...")
        
        rate = rospy.Rate(self.publish_rate)
        self.start_time = rospy.get_time()
        
        # Start plotting thread
        if self.plotting_enabled:
            self.plot_thread = threading.Thread(target=self.plot_loop)
            self.plot_thread.daemon = True
            self.plot_thread.start()
        
        try:
            while not rospy.is_shutdown():
                current_time = rospy.get_time()
                t = current_time - self.start_time
                
                # Stop after test duration
                if t > self.test_duration:
                    rospy.loginfo("Test completed!")
                    break
                
                # Generate target position
                target_pos = self.generate_safe_trajectory(t)
                
                # Create and publish target message
                target_msg = JointState()
                target_msg.header = Header()
                target_msg.header.stamp = rospy.Time.now()
                target_msg.header.frame_id = ""
                target_msg.name = self.joint_names
                target_msg.position = target_pos.tolist()
                target_msg.velocity = [0.0] * 7
                target_msg.effort = [0.0] * 7
                
                self.target_pub.publish(target_msg)
                
                # Update tracking error
                self.update_tracking_error(target_pos, self.current_positions, t)
                
                # Print status every 2 seconds
                if int(t * 2) != int((t - 1/self.publish_rate) * 2):
                    error = target_pos - self.current_positions
                    self.print_status(t, target_pos, error)
                
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("Test interrupted by user")
        
        # Final statistics
        self.print_final_statistics()
    
    def plot_loop(self):
        """Separate thread for plotting"""
        plt.ion()
        fig, axes = plt.subplots(3, 1, figsize=(12,10))

        while not rospy.is_shutdown() and self.plotting_enabled:
            self.plot_results(axes)
            time.sleep(0.1)  # Update plot at 10Hz
    
    def print_final_statistics(self):
        """Print final test statistics"""
        print("\n" + "="*80)
        print("FINAL TEST STATISTICS")
        print("="*80)
        
        if len(self.error_history) == 0:
            print("No data collected!")
            return
        
        all_errors = np.array(list(self.error_history))
        
        print(f"Test duration: {len(all_errors) / self.publish_rate:.2f} seconds")
        print(f"Data points: {len(all_errors)}")
        
        print("\nPer-joint statistics (rad):")
        print("Joint |   Max Error |  RMS Error  |  Max Error (deg)")
        print("------|-------------|-------------|------------------")
        for i in range(7):
            max_err = np.max(np.abs(all_errors[:, i]))
            rms_err = np.sqrt(np.mean(all_errors[:, i]**2))
            max_err_deg = np.degrees(max_err)
            print(f"  {i+1}   |   {max_err:8.4f}  |  {rms_err:8.4f}  |     {max_err_deg:8.2f}")
        
        print(f"\nOverall maximum error: {np.max(np.abs(all_errors)):.4f} rad ({np.degrees(np.max(np.abs(all_errors))):.2f} deg)")
        print(f"Overall RMS error: {np.sqrt(np.mean(all_errors**2)):.4f} rad ({np.degrees(np.sqrt(np.mean(all_errors**2))):.2f} deg)")
        
        # Performance assessment
        max_total_error = np.max(np.abs(all_errors))
        if max_total_error < 0.005:  # < 0.3 degrees
            print("\n✅ EXCELLENT: Very low tracking error")
        elif max_total_error < 0.01:  # < 0.6 degrees
            print("\n✅ GOOD: Acceptable tracking error")
        elif max_total_error < 0.02:  # < 1.1 degrees
            print("\n⚠️  FAIR: Moderate tracking error")
        else:
            print("\n❌ POOR: High tracking error - check controller parameters")

if __name__ == '__main__':
    try:
        # You can modify these parameters
        publisher = FrankaTestTrajectoryPublisher()
        
        # Customize test parameters
        publisher.trajectory_type = 'sine_wave'  # 'sine_wave', 'step', 'circular', 'figure_eight'
        publisher.amplitude = 0.2  # Reduced for safety
        publisher.frequency = 0.15  # Slightly faster
        publisher.test_duration = 20.0  # 20 second test
        publisher.plotting_enabled = True  # Set to False if no display
        
        rospy.loginfo(f"Test configuration:")
        rospy.loginfo(f"  Trajectory: {publisher.trajectory_type}")
        rospy.loginfo(f"  Amplitude: {publisher.amplitude} rad ({np.degrees(publisher.amplitude):.1f} deg)")
        rospy.loginfo(f"  Frequency: {publisher.frequency} Hz")
        rospy.loginfo(f"  Duration: {publisher.test_duration} s")
        
        publisher.run_test()
        
        # Keep node alive for final plotting
        rospy.loginfo("Test completed. Press Ctrl+C to exit...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass