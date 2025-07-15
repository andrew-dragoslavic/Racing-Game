import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt

class CarRacingPreprocessor:
    def __init__(self, img_size = (84,84), crop_region = (0, 84, 6, 90)):
        self.img_size = img_size
        self.crop_region = crop_region
    
    def preprocess(self, observation):
        top, bottom, left, right = self.crop_region
        cropped = observation[top:bottom, left:right]

        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(gray, self.img_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0

        return normalized

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames = 4, skip_frames = 4):
        super().__init__(env)
        self.num_frames = num_frames
        self.skip_frames = skip_frames
        self.preprocessor = CarRacingPreprocessor()

        self.frames = deque(maxlen=num_frames)

        self.observation_space = gym.spaces.Box(
            high=1, low=0, shape=(num_frames, 84, 84), dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for _ in range(50):
            obs, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        
        processed_frame = self.preprocessor.preprocess(obs)
        for _ in range(self.num_frames):
            self.frames.append(processed_frame)
        
        return self._get_observation(), info
    
    def step(self, action):
        total_reward = 0

        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        processed_frame = self.preprocessor.preprocess(obs)
        self.frames.append(processed_frame)

        return self._get_observation(), total_reward, terminated, truncated, info
    
    def _get_observation(self):
        return np.array(list(self.frames), dtype=np.float32)
    
    def check_road_visibility(self, observation):
        """Oliver's road visibility check"""
        # Convert to RGB if needed
        if len(observation.shape) == 4:  # Batch dimension
            rgb_frame = observation[0]
        else:
            rgb_frame = observation
            
        # Take last frame from stack
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[0] == 4:
            rgb_frame = rgb_frame[-1]  # Last frame
            
        # Convert to 0-255 range if normalized
        if rgb_frame.max() <= 1.0:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)
            
        # Create mask to detect road (gray pixels)
        mask = cv2.inRange(rgb_frame, np.array([100, 100, 100]), np.array([150, 150, 150]))
        
        # Return True if road is visible
        return np.any(mask == 255)
    
def test_preprocessing():
    """Test the preprocessing pipeline"""
    
    print("=== Testing Preprocessing Pipeline ===")
    
    # Create environment with preprocessing
    env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=False)
    env = FrameStackWrapper(env, num_frames=4, skip_frames=4)
    
    print(f"Original observation space: Box(0, 255, (96, 96, 3), uint8)")
    print(f"Processed observation space: {env.observation_space}")
    
    # Reset and get initial observation
    obs, info = env.reset()
    print(f"Processed observation shape: {obs.shape}")
    print(f"Processed observation dtype: {obs.dtype}")
    print(f"Processed observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Take a few steps to see frame changes
    print("\nTaking steps to see frame evolution...")
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: Reward={reward:.3f}, Shape={obs.shape}")
        
        if terminated or truncated:
            obs, info = env.reset()
            print("Episode ended, reset environment")
    
    # Visualize the frame stack
    plt.figure(figsize=(15, 4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(obs[i], cmap='gray')
        plt.title(f'Frame {i+1} (t-{3-i})')
        plt.axis('off')
    
    plt.suptitle('Stacked Frames (Most Recent on Right)')
    plt.tight_layout()
    plt.savefig('frame_stacking_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved frame stacking visualization to 'frame_stacking_visualization.png'")
    
    env.close()

def compare_before_after():
    """Compare raw vs processed observations"""
    
    print("\n=== Before/After Preprocessing Comparison ===")
    
    # Raw environment
    raw_env = gym.make('CarRacing-v3', render_mode='rgb_array', continuous=False)
    raw_obs, _ = raw_env.reset()
    
    # Take some steps to get interesting frame
    for _ in range(60):
        raw_obs, _, terminated, truncated, _ = raw_env.step(3)  # gas
        if terminated or truncated:
            raw_obs, _ = raw_env.reset()
    
    # Process the frame
    preprocessor = CarRacingPreprocessor()
    processed_obs = preprocessor.preprocess(raw_obs)
    
    # Display comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(raw_obs)
    plt.title(f'Raw Observation\n{raw_obs.shape}')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(processed_obs, cmap='gray')
    plt.title(f'Processed Observation\n{processed_obs.shape}')
    plt.axis('off')
    
    # Show size reduction
    raw_size = raw_obs.size * raw_obs.itemsize
    processed_size = processed_obs.size * processed_obs.itemsize
    reduction = (1 - processed_size / raw_size) * 100
    
    plt.subplot(1, 3, 3)
    plt.bar(['Raw', 'Processed'], [raw_size, processed_size], color=['red', 'green'])
    plt.title(f'Memory Usage\n{reduction:.1f}% reduction')
    plt.ylabel('Bytes')
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved preprocessing comparison to 'preprocessing_comparison.png'")
    
    print(f"Raw size: {raw_obs.shape} = {raw_size:,} bytes")
    print(f"Processed size: {processed_obs.shape} = {processed_size:,} bytes")
    print(f"Memory reduction: {reduction:.1f}%")
    
    raw_env.close()

if __name__ == "__main__":
    test_preprocessing()
    compare_before_after()   