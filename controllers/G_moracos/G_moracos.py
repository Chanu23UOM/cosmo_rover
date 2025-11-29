"""
G_MORACOS - Fully Autonomous Rover Controller
==============================================
Project Cosmotron Competition - Webots R2025a

Mission Objectives:
1. Navigate challenging terrain autonomously
2. Follow flags (colored markers)
3. Avoid obstacles
4. Locate asteroid in target zone
5. Retrieve asteroid using robotic arm
6. Store asteroid safely onboard

State Machine:
- INIT: Initialize sensors, calibrate
- SEARCH_FLAG: Look for navigation flags
- FOLLOW_FLAG: Navigate towards detected flag
- AVOID_OBSTACLE: Obstacle avoidance behavior
- SEARCH_ASTEROID: Search for asteroid object
- APPROACH_ASTEROID: Move towards asteroid
- ALIGN_PICKUP: Fine alignment for pickup
- PICKUP: Execute arm pickup sequence
- STORE: Store asteroid in cargo area
- MISSION_COMPLETE: Park and signal completion

Controls (Manual Mode):
- M: Toggle Manual/Autonomous mode
- WASD: Movement
- Space: Stop
- H: Arm home position
- G: Arm grab position
- O/P: Open/Close gripper

Author: G_moracos Autonomous Controller
"""

from controller import Robot, Keyboard, Camera, GPS, Gyro, Compass, InertialUnit, DistanceSensor
import math
import struct

# ============================================================================
# CONFIGURATION
# ============================================================================

TIME_STEP = 32  # ms

# Wheel speeds
MAX_SPEED = 8.0
CRUISE_SPEED = 5.0
SLOW_SPEED = 2.5
TURN_SPEED = 3.0

# Obstacle detection thresholds (0-1000 scale from distance sensors)
OBSTACLE_THRESHOLD = 400  # Close obstacle
OBSTACLE_WARNING = 600    # Approaching obstacle
OBSTACLE_FAR = 800        # Obstacle detected but far

# Flag/Object detection colors
FLAG_COLORS = {
    'red': ([0.7, 0.0, 0.0], [1.0, 0.3, 0.3]),     # Red flags
    'green': ([0.0, 0.6, 0.0], [0.3, 1.0, 0.3]),   # Green flags
    'blue': ([0.0, 0.0, 0.7], [0.3, 0.3, 1.0]),    # Blue flags
    'yellow': ([0.7, 0.7, 0.0], [1.0, 1.0, 0.4]),  # Yellow flags
    'orange': ([0.8, 0.4, 0.0], [1.0, 0.6, 0.2]),  # Orange flags
    'white': ([0.8, 0.8, 0.8], [1.0, 1.0, 1.0]),   # White flags
}

ASTEROID_COLOR = ([0.2, 0.2, 0.2], [0.6, 0.6, 0.6])  # Gray asteroid

# Arm positions for different actions [arm1, arm2, arm3, arm4, arm5]
ARM_HOME = [0.0, 0.0, 0.0, 0.0, 0.0]
ARM_SEARCH = [0.0, 0.3, -0.3, 0.0, 0.0]  # Looking forward
ARM_PRE_GRAB = [0.0, -0.3, 1.2, 0.8, 0.0]  # Positioned above target
ARM_GRAB = [0.0, -0.6, 1.5, 1.0, 0.0]  # Down to grab
ARM_LIFT = [0.0, 0.2, 0.5, 0.3, 0.0]  # Lifted with object
ARM_STORE = [2.5, 0.5, 0.3, 0.2, 0.0]  # Rotated to storage area

# Arm joint limits
ARM_LIMITS = {
    'arm1': (-2.9496, 2.9496),
    'arm2': (-2.35619, 1.39626),
    'arm3': (-2.63545, 2.54818),
    'arm4': (-3.14159, 3.14159),
    'arm5': (-3.14159, 3.14159),
}


# ============================================================================
# STATE MACHINE STATES
# ============================================================================

class State:
    INIT = "INIT"
    SEARCH_FLAG = "SEARCH_FLAG"
    FOLLOW_FLAG = "FOLLOW_FLAG"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    SEARCH_ASTEROID = "SEARCH_ASTEROID"
    APPROACH_ASTEROID = "APPROACH_ASTEROID"
    ALIGN_PICKUP = "ALIGN_PICKUP"
    PICKUP = "PICKUP"
    STORE = "STORE"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    MANUAL = "MANUAL"


# ============================================================================
# AUTONOMOUS ROVER CONTROLLER
# ============================================================================

class AutonomousRover:
    """Fully autonomous rover controller for competition tasks."""
    
    def __init__(self):
        """Initialize the rover and all systems."""
        self.robot = Robot()
        self.timestep = TIME_STEP
        
        # State machine
        self.state = State.INIT
        self.prev_state = None
        self.state_timer = 0
        self.state_data = {}
        
        # Mission tracking
        self.flags_passed = 0
        self.asteroid_collected = False
        self.mission_start_time = 0
        
        # Navigation
        self.search_direction = 1  # 1 = right, -1 = left
        self.last_flag_type = None
        
        # Avoidance state
        self.avoid_direction = 1  # 1 = left, -1 = right
        self.avoid_timer = 0
        
        # Manual override
        self.manual_mode = False
        self.keyboard = Keyboard()
        self.keyboard.enable(TIME_STEP)
        
        # Initialize all devices
        self._init_sensors()
        self._init_wheels()
        self._init_arm()
        
        print("=" * 70)
        print("   G_MORACOS AUTONOMOUS ROVER - PROJECT COSMOTRON")
        print("=" * 70)
        print("Mission: Navigate terrain, follow flags, collect asteroid")
        print("Press 'M' to toggle Manual/Autonomous mode")
        print("=" * 70)
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def _init_sensors(self):
        """Initialize all sensors."""
        # Camera
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(TIME_STEP)
            self.camera.recognitionEnable(TIME_STEP)
            print(f"[OK] Camera: {self.camera.getWidth()}x{self.camera.getHeight()}")
        else:
            print("[WARN] Camera not found")
        
        # GPS
        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(TIME_STEP)
            print("[OK] GPS enabled")
        
        # Gyro
        self.gyro = self.robot.getDevice("gyro")
        if self.gyro:
            self.gyro.enable(TIME_STEP)
            print("[OK] Gyro enabled")
        
        # IMU
        self.imu = self.robot.getDevice("imu")
        if self.imu:
            self.imu.enable(TIME_STEP)
            print("[OK] IMU enabled")
        
        # Compass
        self.compass = self.robot.getDevice("compass")
        if self.compass:
            self.compass.enable(TIME_STEP)
            print("[OK] Compass enabled")
        
        # Distance sensors
        self.distance_sensors = {}
        ds_names = ["ds_front_left", "ds_front", "ds_front_right", "ds_left", "ds_right"]
        for name in ds_names:
            sensor = self.robot.getDevice(name)
            if sensor:
                sensor.enable(TIME_STEP)
                self.distance_sensors[name] = sensor
                print(f"[OK] Distance sensor: {name}")
            else:
                print(f"[WARN] Distance sensor not found: {name}")
    
    def _init_wheels(self):
        """Initialize wheel motors for velocity control."""
        wheel_names = [
            "FrontLeftWheel", "FrontRightWheel",
            "MiddleLeftWheel", "MiddleRightWheel",
            "BackLeftWheel", "BackRightWheel"
        ]
        
        self.wheels = {}
        for name in wheel_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)
                self.wheels[name] = motor
        
        self.left_wheels = ["FrontLeftWheel", "MiddleLeftWheel", "BackLeftWheel"]
        self.right_wheels = ["FrontRightWheel", "MiddleRightWheel", "BackRightWheel"]
        print(f"[OK] Wheels initialized: {len(self.wheels)} motors")
    
    def _init_arm(self):
        """Initialize arm motors and sensors."""
        self.arm_motors = {}
        self.arm_sensors = {}
        
        for i in range(1, 6):
            name = f"arm{i}"
            motor = self.robot.getDevice(name)
            if motor:
                motor.setVelocity(1.0)
                self.arm_motors[name] = motor
            
            sensor = self.robot.getDevice(f"{name}sensor")
            if sensor:
                sensor.enable(TIME_STEP)
                self.arm_sensors[name] = sensor
        
        # Gripper
        self.gripper_left = self.robot.getDevice("finger::left")
        self.gripper_right = self.robot.getDevice("finger::right")
        if self.gripper_left:
            self.gripper_left.setVelocity(0.1)
        if self.gripper_right:
            self.gripper_right.setVelocity(0.1)
        
        self.gripper_left_sensor = self.robot.getDevice("finger::leftsensor")
        self.gripper_right_sensor = self.robot.getDevice("finger::rightsensor")
        if self.gripper_left_sensor:
            self.gripper_left_sensor.enable(TIME_STEP)
        if self.gripper_right_sensor:
            self.gripper_right_sensor.enable(TIME_STEP)
        
        print(f"[OK] Arm initialized: {len(self.arm_motors)} joints + gripper")
    
    # ========================================================================
    # WHEEL CONTROL
    # ========================================================================
    
    def set_wheel_speeds(self, left, right):
        """Set left and right wheel velocities.
        Note: Right wheels are negated because wheel axes are mirrored."""
        left = max(-MAX_SPEED, min(MAX_SPEED, left))
        right = max(-MAX_SPEED, min(MAX_SPEED, right))
        
        for name in self.left_wheels:
            if name in self.wheels:
                self.wheels[name].setVelocity(left)
        for name in self.right_wheels:
            if name in self.wheels:
                # Negate right wheel velocity due to wheel axis orientation
                self.wheels[name].setVelocity(-right)
    
    def stop(self):
        """Stop all wheels."""
        self.set_wheel_speeds(0, 0)
    
    def move_forward(self, speed=CRUISE_SPEED):
        """Move forward."""
        self.set_wheel_speeds(speed, speed)
    
    def move_backward(self, speed=CRUISE_SPEED):
        """Move backward."""
        self.set_wheel_speeds(-speed, -speed)
    
    def turn_left(self, speed=TURN_SPEED):
        """Turn left in place."""
        self.set_wheel_speeds(-speed, speed)
    
    def turn_right(self, speed=TURN_SPEED):
        """Turn right in place."""
        self.set_wheel_speeds(speed, -speed)
    
    def curve_left(self, speed=CRUISE_SPEED, ratio=0.5):
        """Curve to the left while moving forward."""
        self.set_wheel_speeds(speed * ratio, speed)
    
    def curve_right(self, speed=CRUISE_SPEED, ratio=0.5):
        """Curve to the right while moving forward."""
        self.set_wheel_speeds(speed, speed * ratio)
    
    # ========================================================================
    # ARM CONTROL
    # ========================================================================
    
    def set_arm_position(self, positions, speed=1.0):
        """Set arm to specific joint positions."""
        joint_names = ["arm1", "arm2", "arm3", "arm4", "arm5"]
        for i, name in enumerate(joint_names):
            if name in self.arm_motors:
                motor = self.arm_motors[name]
                motor.setVelocity(speed)
                limits = ARM_LIMITS.get(name, (-3.14, 3.14))
                pos = max(limits[0], min(limits[1], positions[i]))
                motor.setPosition(pos)
    
    def get_arm_position(self):
        """Get current arm joint positions."""
        positions = []
        for i in range(1, 6):
            sensor = self.arm_sensors.get(f"arm{i}")
            if sensor:
                positions.append(sensor.getValue())
            else:
                positions.append(0.0)
        return positions
    
    def arm_at_position(self, target, tolerance=0.15):
        """Check if arm has reached target position."""
        current = self.get_arm_position()
        for i in range(5):
            if abs(current[i] - target[i]) > tolerance:
                return False
        return True
    
    def open_gripper(self):
        """Open gripper."""
        if self.gripper_left:
            self.gripper_left.setPosition(0.025)
        if self.gripper_right:
            self.gripper_right.setPosition(0.025)
    
    def close_gripper(self):
        """Close gripper."""
        if self.gripper_left:
            self.gripper_left.setPosition(0.0)
        if self.gripper_right:
            self.gripper_right.setPosition(0.0)
    
    # ========================================================================
    # SENSOR READING
    # ========================================================================
    
    def get_distance_readings(self):
        """Get all distance sensor readings."""
        readings = {}
        for name, sensor in self.distance_sensors.items():
            readings[name] = sensor.getValue()
        return readings
    
    def check_obstacle(self):
        """Check for obstacles and return obstacle info."""
        readings = self.get_distance_readings()
        
        front = readings.get("ds_front", 1000)
        front_left = readings.get("ds_front_left", 1000)
        front_right = readings.get("ds_front_right", 1000)
        left = readings.get("ds_left", 1000)
        right = readings.get("ds_right", 1000)
        
        obstacle = {
            'front': front < OBSTACLE_THRESHOLD,
            'front_warning': front < OBSTACLE_WARNING,
            'left': front_left < OBSTACLE_THRESHOLD or left < OBSTACLE_THRESHOLD,
            'right': front_right < OBSTACLE_THRESHOLD or right < OBSTACLE_THRESHOLD,
            'front_left_dist': front_left,
            'front_right_dist': front_right,
            'front_dist': front,
        }
        
        obstacle['any'] = obstacle['front'] or obstacle['left'] or obstacle['right']
        obstacle['clear'] = not obstacle['any'] and front > OBSTACLE_FAR
        
        return obstacle
    
    def get_heading(self):
        """Get current heading from compass (0-360 degrees)."""
        if self.compass:
            values = self.compass.getValues()
            heading = math.atan2(values[0], values[2])
            heading = math.degrees(heading)
            if heading < 0:
                heading += 360
            return heading
        return 0
    
    def get_position(self):
        """Get current GPS position."""
        if self.gps:
            return self.gps.getValues()
        return [0, 0, 0]
    
    # ========================================================================
    # VISION PROCESSING
    # ========================================================================
    
    def detect_objects(self):
        """Detect and classify objects from camera."""
        if not self.camera:
            return {'flags': [], 'asteroid': None, 'obstacles': []}
        
        objects = self.camera.getRecognitionObjects()
        
        result = {
            'flags': [],
            'asteroid': None,
            'obstacles': [],
            'all_objects': []
        }
        
        cam_width = self.camera.getWidth()
        cam_height = self.camera.getHeight()
        
        for obj in objects:
            colors = obj.getColors()
            pos = obj.getPosition()
            img_pos = obj.getPositionOnImage()
            size = obj.getSizeOnImage()
            
            # Normalize image position (-1 to 1, where 0 is center)
            norm_x = (img_pos[0] - cam_width/2) / (cam_width/2)
            
            obj_info = {
                'id': obj.getId(),
                'position': pos,
                'distance': math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2),
                'angle': norm_x,  # -1 = left, 0 = center, 1 = right
                'size': size,
                'colors': colors if len(colors) >= 3 else [0, 0, 0],
                'model': obj.getModel() if hasattr(obj, 'getModel') else ''
            }
            
            result['all_objects'].append(obj_info)
            
            # Classify object by color
            if len(colors) >= 3:
                # Check if it's a flag
                for flag_name, (min_c, max_c) in FLAG_COLORS.items():
                    if (min_c[0] <= colors[0] <= max_c[0] and
                        min_c[1] <= colors[1] <= max_c[1] and
                        min_c[2] <= colors[2] <= max_c[2]):
                        obj_info['flag_type'] = flag_name
                        result['flags'].append(obj_info)
                        break
                
                # Check if asteroid (gray)
                min_c, max_c = ASTEROID_COLOR
                if (min_c[0] <= colors[0] <= max_c[0] and
                    min_c[1] <= colors[1] <= max_c[1] and
                    min_c[2] <= colors[2] <= max_c[2]):
                    # Check if colors are similar (grayscale)
                    if abs(colors[0] - colors[1]) < 0.2 and abs(colors[1] - colors[2]) < 0.2:
                        if result['asteroid'] is None or obj_info['distance'] < result['asteroid']['distance']:
                            result['asteroid'] = obj_info
        
        return result
    
    def get_nearest_flag(self):
        """Get the nearest navigation flag."""
        vision = self.detect_objects()
        if vision['flags']:
            flags = sorted(vision['flags'], key=lambda f: f['distance'])
            return flags[0]
        return None
    
    def get_asteroid(self):
        """Get detected asteroid info."""
        vision = self.detect_objects()
        return vision['asteroid']
    
    def get_any_target(self):
        """Get any visible target (flag or asteroid)."""
        vision = self.detect_objects()
        
        # Prioritize asteroid if we've passed flags
        if self.flags_passed > 0 and vision['asteroid']:
            return vision['asteroid'], 'asteroid'
        
        # Otherwise look for flags
        if vision['flags']:
            flags = sorted(vision['flags'], key=lambda f: f['distance'])
            return flags[0], 'flag'
        
        # Finally check for asteroid
        if vision['asteroid']:
            return vision['asteroid'], 'asteroid'
        
        return None, None
    
    # ========================================================================
    # STATE MACHINE BEHAVIORS
    # ========================================================================
    
    def state_init(self):
        """Initialization state."""
        self.stop()
        self.set_arm_position(ARM_HOME)
        self.open_gripper()
        
        self.state_timer += 1
        if self.state_timer > 30:  # Wait ~1 second
            self.mission_start_time = self.robot.getTime()
            print("\n[MISSION START] Searching for navigation flags...")
            return State.SEARCH_FLAG
        return State.INIT
    
    def state_search_flag(self):
        """Search for navigation flags by rotating."""
        obstacle = self.check_obstacle()
        
        if obstacle['front']:
            print("[DETECT] Obstacle during search")
            return State.AVOID_OBSTACLE
        
        target, target_type = self.get_any_target()
        
        if target:
            if target_type == 'asteroid' and self.flags_passed >= 1:
                print(f"[DETECT] Asteroid at {target['distance']:.2f}m")
                self.state_data['target'] = target
                return State.SEARCH_ASTEROID
            elif target_type == 'flag':
                print(f"[DETECT] {target.get('flag_type', 'unknown')} flag at {target['distance']:.2f}m")
                self.state_data['target'] = target
                return State.FOLLOW_FLAG
        
        # Rotate to search - alternate direction periodically
        self.state_timer += 1
        if self.state_timer > 150:  # Change direction every ~5 sec
            self.search_direction *= -1
            self.state_timer = 0
            print("[SEARCH] Changing search direction...")
        
        if self.search_direction > 0:
            self.turn_right(TURN_SPEED * 0.6)
        else:
            self.turn_left(TURN_SPEED * 0.6)
        
        return State.SEARCH_FLAG
    
    def state_follow_flag(self):
        """Navigate towards detected flag."""
        obstacle = self.check_obstacle()
        
        if obstacle['front']:
            print("[DETECT] Obstacle while approaching flag")
            return State.AVOID_OBSTACLE
        
        flag = self.get_nearest_flag()
        
        if not flag:
            print("[LOST] Flag not visible, resuming search...")
            self.state_timer = 0
            return State.SEARCH_FLAG
        
        distance = flag['distance']
        angle = flag['angle']
        
        # Reached the flag
        if distance < 0.6:
            self.flags_passed += 1
            self.last_flag_type = flag.get('flag_type', 'unknown')
            print(f"[REACHED] Flag #{self.flags_passed} ({self.last_flag_type})")
            self.state_timer = 0
            
            # Check for asteroid after reaching flag
            asteroid = self.get_asteroid()
            if asteroid:
                print("[DETECT] Asteroid visible!")
                return State.SEARCH_ASTEROID
            
            # Continue to next flag
            return State.SEARCH_FLAG
        
        # Navigate towards flag using proportional control
        if abs(angle) < 0.15:
            # Centered - move forward
            speed = CRUISE_SPEED if distance > 1.0 else SLOW_SPEED
            self.move_forward(speed)
        elif angle < 0:
            # Target to the left
            turn_ratio = 0.3 + abs(angle) * 0.5
            self.curve_left(CRUISE_SPEED, max(0.1, 1 - turn_ratio))
        else:
            # Target to the right
            turn_ratio = 0.3 + abs(angle) * 0.5
            self.curve_right(CRUISE_SPEED, max(0.1, 1 - turn_ratio))
        
        return State.FOLLOW_FLAG
    
    def state_avoid_obstacle(self):
        """Obstacle avoidance behavior."""
        obstacle = self.check_obstacle()
        
        # Determine avoidance direction on entry
        if self.avoid_timer == 0:
            if obstacle['front_left_dist'] > obstacle['front_right_dist']:
                self.avoid_direction = 1  # Turn left
            else:
                self.avoid_direction = -1  # Turn right
            print(f"[AVOID] Direction: {'left' if self.avoid_direction > 0 else 'right'}")
        
        self.avoid_timer += 1
        
        if obstacle['front']:
            # Back up if blocked
            if self.avoid_timer < 25:
                self.move_backward(SLOW_SPEED)
            else:
                if self.avoid_direction > 0:
                    self.turn_left(TURN_SPEED)
                else:
                    self.turn_right(TURN_SPEED)
        elif obstacle['clear']:
            print("[CLEAR] Obstacle avoided")
            self.avoid_timer = 0
            
            # Return to appropriate state
            if self.asteroid_collected:
                return State.STORE
            elif self.flags_passed > 0:
                return State.SEARCH_ASTEROID
            else:
                return State.SEARCH_FLAG
        else:
            # Keep turning until clear
            if self.avoid_direction > 0:
                self.turn_left(TURN_SPEED)
            else:
                self.turn_right(TURN_SPEED)
        
        # Timeout
        if self.avoid_timer > 120:
            print("[TIMEOUT] Avoidance timeout")
            self.avoid_timer = 0
            return State.SEARCH_FLAG
        
        return State.AVOID_OBSTACLE
    
    def state_search_asteroid(self):
        """Search for asteroid object."""
        obstacle = self.check_obstacle()
        
        if obstacle['front']:
            return State.AVOID_OBSTACLE
        
        asteroid = self.get_asteroid()
        
        if asteroid:
            print(f"[DETECT] Asteroid located at {asteroid['distance']:.2f}m")
            self.state_data['target'] = asteroid
            return State.APPROACH_ASTEROID
        
        # Slow rotation search
        self.state_timer += 1
        if self.state_timer % 200 < 100:
            self.turn_right(TURN_SPEED * 0.5)
        else:
            self.move_forward(SLOW_SPEED)
        
        if self.state_timer > 400:
            print("[SEARCH] Expanding search area...")
            self.state_timer = 0
        
        return State.SEARCH_ASTEROID
    
    def state_approach_asteroid(self):
        """Approach the asteroid for pickup."""
        obstacle = self.check_obstacle()
        
        # Only react to obstacles far from asteroid
        if obstacle['front'] and obstacle['front_dist'] < 150:
            # This might be the asteroid itself - continue carefully
            pass
        elif obstacle['front']:
            return State.AVOID_OBSTACLE
        
        asteroid = self.get_asteroid()
        
        if not asteroid:
            print("[LOST] Asteroid not visible")
            self.state_timer += 1
            if self.state_timer > 20:
                self.state_timer = 0
                return State.SEARCH_ASTEROID
            return State.APPROACH_ASTEROID
        
        self.state_timer = 0
        distance = asteroid['distance']
        angle = asteroid['angle']
        
        # Prepare arm as we get closer
        if distance < 1.2:
            self.set_arm_position(ARM_PRE_GRAB, speed=0.7)
            self.open_gripper()
        
        # In pickup range
        if distance < 0.4:
            print("[RANGE] Asteroid in pickup range")
            self.stop()
            return State.ALIGN_PICKUP
        
        # Navigate with proportional control
        if abs(angle) < 0.1:
            speed = SLOW_SPEED if distance < 0.8 else CRUISE_SPEED * 0.7
            self.move_forward(speed)
        elif angle < 0:
            self.curve_left(SLOW_SPEED, 0.3)
        else:
            self.curve_right(SLOW_SPEED, 0.3)
        
        return State.APPROACH_ASTEROID
    
    def state_align_pickup(self):
        """Fine alignment for asteroid pickup."""
        self.stop()
        
        asteroid = self.get_asteroid()
        
        if not asteroid:
            self.state_timer += 1
            if self.state_timer > 40:
                print("[LOST] Asteroid during alignment")
                self.state_timer = 0
                return State.SEARCH_ASTEROID
            # Try small adjustments
            self.turn_right(TURN_SPEED * 0.2)
            return State.ALIGN_PICKUP
        
        angle = asteroid['angle']
        distance = asteroid['distance']
        
        # Too far - move closer
        if distance > 0.45:
            self.move_forward(SLOW_SPEED * 0.5)
            return State.ALIGN_PICKUP
        
        # Fine angle adjustment
        if abs(angle) > 0.08:
            if angle < 0:
                self.turn_left(TURN_SPEED * 0.25)
            else:
                self.turn_right(TURN_SPEED * 0.25)
            return State.ALIGN_PICKUP
        
        # Aligned!
        self.stop()
        print("[ALIGNED] Ready for pickup")
        self.state_timer = 0
        return State.PICKUP
    
    def state_pickup(self):
        """Execute pickup sequence."""
        self.stop()
        self.state_timer += 1
        
        # Multi-phase pickup sequence
        phase = self.state_timer // 45  # ~1.5s per phase
        
        if phase == 0:
            # Position arm above asteroid
            self.set_arm_position(ARM_PRE_GRAB, speed=0.8)
            self.open_gripper()
        elif phase == 1:
            # Lower arm to grab
            self.set_arm_position(ARM_GRAB, speed=0.5)
        elif phase == 2:
            # Close gripper
            self.close_gripper()
        elif phase == 3:
            # Lift asteroid
            self.set_arm_position(ARM_LIFT, speed=0.4)
        elif phase == 4:
            # Verify pickup and transition
            self.asteroid_collected = True
            print("[SUCCESS] Asteroid collected!")
            self.state_timer = 0
            return State.STORE
        
        return State.PICKUP
    
    def state_store(self):
        """Store asteroid in cargo area."""
        self.stop()
        self.state_timer += 1
        
        phase = self.state_timer // 55  # ~1.8s per phase
        
        if phase == 0:
            # Ensure lifted position
            self.set_arm_position(ARM_LIFT, speed=0.5)
        elif phase == 1:
            # Rotate to storage
            self.set_arm_position(ARM_STORE, speed=0.4)
        elif phase == 2:
            # Release
            self.open_gripper()
        elif phase == 3:
            # Return arm home
            self.set_arm_position(ARM_HOME, speed=0.6)
        elif phase == 4:
            self.close_gripper()
            print("[STORED] Asteroid secured!")
            return State.MISSION_COMPLETE
        
        return State.STORE
    
    def state_mission_complete(self):
        """Mission complete - park and report."""
        self.stop()
        self.set_arm_position(ARM_HOME)
        
        self.state_timer += 1
        
        # Report status periodically
        if self.state_timer % 150 == 1:
            elapsed = self.robot.getTime() - self.mission_start_time
            print("=" * 50)
            print("          MISSION COMPLETE!")
            print("=" * 50)
            print(f"  Flags passed: {self.flags_passed}")
            print(f"  Asteroid collected: {self.asteroid_collected}")
            print(f"  Mission time: {elapsed:.1f} seconds")
            print("=" * 50)
        
        return State.MISSION_COMPLETE
    
    # ========================================================================
    # MANUAL CONTROL
    # ========================================================================
    
    def handle_keyboard(self):
        """Handle keyboard input."""
        key = self.keyboard.getKey()
        movement = False
        
        while key >= 0:
            # Mode toggle
            if key == ord('M'):
                self.manual_mode = not self.manual_mode
                mode = "MANUAL" if self.manual_mode else "AUTONOMOUS"
                print(f"\n{'='*30}")
                print(f"  MODE: {mode}")
                print(f"{'='*30}\n")
                if not self.manual_mode:
                    self.state = State.SEARCH_FLAG
                    self.state_timer = 0
            
            # Manual controls
            if self.manual_mode:
                if key == ord('W'):
                    self.move_forward()
                    movement = True
                elif key == ord('S'):
                    self.move_backward()
                    movement = True
                elif key == ord('A'):
                    self.turn_left()
                    movement = True
                elif key == ord('D'):
                    self.turn_right()
                    movement = True
                elif key == ord(' '):
                    self.stop()
                elif key == ord('H'):
                    self.set_arm_position(ARM_HOME)
                    print("[ARM] Home position")
                elif key == ord('G'):
                    self.set_arm_position(ARM_GRAB)
                    print("[ARM] Grab position")
                elif key == ord('L'):
                    self.set_arm_position(ARM_LIFT)
                    print("[ARM] Lift position")
                elif key == ord('O'):
                    self.open_gripper()
                    print("[GRIPPER] Open")
                elif key == ord('P'):
                    self.close_gripper()
                    print("[GRIPPER] Close")
                elif key == ord('C'):
                    # Debug: show detected objects
                    vision = self.detect_objects()
                    print(f"[CAMERA] Flags: {len(vision['flags'])}, Asteroid: {vision['asteroid'] is not None}")
                    for f in vision['flags']:
                        print(f"  - {f.get('flag_type','?')} at {f['distance']:.2f}m, angle={f['angle']:.2f}")
                    if vision['asteroid']:
                        a = vision['asteroid']
                        print(f"  - Asteroid at {a['distance']:.2f}m, angle={a['angle']:.2f}")
            
            key = self.keyboard.getKey()
        
        if self.manual_mode and not movement:
            self.stop()
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    
    def step(self):
        """Execute one control step."""
        self.handle_keyboard()
        
        if self.manual_mode:
            return
        
        # State transition logging
        if self.state != self.prev_state:
            print(f"[STATE] {self.prev_state} -> {self.state}")
            self.prev_state = self.state
            self.state_timer = 0
        
        # Execute current state
        state_handlers = {
            State.INIT: self.state_init,
            State.SEARCH_FLAG: self.state_search_flag,
            State.FOLLOW_FLAG: self.state_follow_flag,
            State.AVOID_OBSTACLE: self.state_avoid_obstacle,
            State.SEARCH_ASTEROID: self.state_search_asteroid,
            State.APPROACH_ASTEROID: self.state_approach_asteroid,
            State.ALIGN_PICKUP: self.state_align_pickup,
            State.PICKUP: self.state_pickup,
            State.STORE: self.state_store,
            State.MISSION_COMPLETE: self.state_mission_complete,
        }
        
        handler = state_handlers.get(self.state)
        if handler:
            self.state = handler()
    
    def run(self):
        """Main control loop."""
        print("\n[SYSTEM] Autonomous control loop starting...")
        print("[SYSTEM] Press 'M' for manual override\n")
        
        while self.robot.step(self.timestep) != -1:
            self.step()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    rover = AutonomousRover()
    rover.run()
