import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class PredictiveEyeMovementModel:
    def __init__(self, 
                 learning_rate=0.1,         # Learning rate for updating history weights
                 initial_cue_weight=0.5,      # Initial weight given to color cue
                 noise_sd=0.2,               # Standard deviation of noise
                 cue_validity={'red': 0.75, 'green': 0.25}):  # Cue validity (probability of right)
        """
        Initialize model parameters:
        - learning_rate: How quickly the model updates history weights
        - initial_cue_weight: Initial reliance on the color cue vs. history
        - noise_sd: Standard deviation of noise in the system
        - cue_validity: Probability of target moving right for each cue
        """
        self.learning_rate = learning_rate
        self.cue_weight = initial_cue_weight
        self.history_weight = 1 - initial_cue_weight
        self.noise_sd = noise_sd
        self.cue_validity = cue_validity
        
        # Internal model parameters
        self.history_bias = 0.5  # Initial history bias (0.5 = no bias)
        self.trial_history = []  # Store outcomes for analysis
        
    def predict_velocity(self, cue):
        """
        Generate predicted velocity based on cue and history:
        - Positive velocity = rightward prediction
        - Negative velocity = leftward prediction
        - Magnitude indicates confidence
        """
        # Convert cue to probability of rightward movement
        cue_prob_right = self.cue_validity[cue]
        
        # Combined prediction (weighted sum of cue and history)
        combined_prob_right = (self.cue_weight * cue_prob_right + 
                               self.history_weight * self.history_bias)
        
        # Convert probability to velocity (-1 to 1 scale)
        # 0.5 probability -> 0 velocity, 1.0 -> +1, 0.0 -> -1
        velocity = (combined_prob_right - 0.5) * 2
        
        # Add noise
        noisy_velocity = velocity + np.random.normal(0, self.noise_sd)
        
        return noisy_velocity, combined_prob_right
    
    def update_history(self, target_direction):
        """
        Update history bias based on observed target direction
        target_direction: 1 for right, 0 for left
        """
        prediction_error = target_direction - self.history_bias
        self.history_bias += self.learning_rate * prediction_error
    
    def update_weights(self, cue, target_direction, adaptation_rate=0.2):
        """
        Update the weights given to cue vs history based on their predictive accuracy
        """
        # Calculate prediction error from cue
        cue_prediction = self.cue_validity[cue]
        cue_error = abs(target_direction - cue_prediction)
        
        # Calculate prediction error from history
        history_error = abs(target_direction - self.history_bias)
        
        # Update weights based on relative errors (smaller error → larger weight)
        if cue_error < history_error:
            # Cue was more accurate, increase its weight
            self.cue_weight = min(1.0, self.cue_weight + adaptation_rate)
        else:
            # History was more accurate, increase its weight
            self.cue_weight = max(0.0, self.cue_weight - adaptation_rate)
            
        self.history_weight = 1 - self.cue_weight
    
    def simulate_trial(self, cue):
        """
        Simulate a single trial with the given cue
        """
        # Generate prediction
        anticipated_velocity, prob_right = self.predict_velocity(cue)
        
        # Determine actual target direction based on cue validity
        target_direction = 1 if np.random.random() < self.cue_validity[cue] else 0
        
        # Update model
        self.update_history(target_direction)
        self.update_weights(cue, target_direction)
        
        # Store trial data
        self.trial_history.append({
            'cue': cue,
            'anticipated_velocity': anticipated_velocity,
            'target_direction': 'right' if target_direction == 1 else 'left',
            'history_bias': self.history_bias,
            'cue_weight': self.cue_weight,
            'history_weight': self.history_weight,
            'prob_right': prob_right
        })
        
        return anticipated_velocity, target_direction
    
    def simulate_experiment(self, n_trials=200, sequence=None):
        """
        Simulate an experiment with n trials
        sequence: predefined sequence of cues (if None, will generate random)
        """
        if sequence is None:
            # Generate random cues
            sequence = np.random.choice(['red', 'green'], size=n_trials)
        
        results = []
        for trial_idx, cue in enumerate(sequence):
            velocity, direction = self.simulate_trial(cue)
            results.append((velocity, direction, cue))
            
        return results
    
    def analyze_results(self, plot=True):
        """
        Analyze and optionally plot the results of the simulation
        """
        df = pd.DataFrame(self.trial_history)
        
        if plot:
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))
            
            # Plot 1: Anticipatory velocity over trials
            axes[0].plot(df['anticipated_velocity'], 'b-', alpha=0.6)
            axes[0].set_title('Anticipatory Eye Movement Velocity')
            axes[0].set_xlabel('Trial')
            axes[0].set_ylabel('Velocity (+ = Right, - = Left)')
            axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add cue color markers
            for i, (idx, row) in enumerate(df.iterrows()):
                color = 'red' if row['cue'] == 'red' else 'green'
                axes[0].scatter(i, row['anticipated_velocity'], color=color, s=20)
            
            # Plot 2: Weight evolution
            axes[1].plot(df['cue_weight'], 'r-', label='Cue Weight')
            axes[1].plot(df['history_weight'], 'b-', label='History Weight')
            axes[1].set_title('Evolution of Weights')
            axes[1].set_xlabel('Trial')
            axes[1].set_ylabel('Weight')
            axes[1].legend()
            
            # Plot 3: History bias
            axes[2].plot(df['history_bias'], 'g-')
            axes[2].set_title('Evolution of History Bias')
            axes[2].set_xlabel('Trial')
            axes[2].set_ylabel('History Bias (Right Probability)')
            axes[2].axhline(y=0.5, color='k', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        # Calculate summary statistics
        summary = {
            'mean_velocity_red': df[df['cue'] == 'red']['anticipated_velocity'].mean(),
            'mean_velocity_green': df[df['cue'] == 'green']['anticipated_velocity'].mean(),
            'final_cue_weight': df['cue_weight'].iloc[-1],
            'final_history_weight': df['history_weight'].iloc[-1],
        }
        
        return df, summary

# Example usage
def run_simulation(manipulation=None):
    """
    Run a simulation with optional experimental manipulation
    
    Manipulations:
    - 'biased_sequence': Use a sequence with biased statistics
    - 'switch_validity': Switch cue validity halfway through
    - 'standard': Standard experiment with random cues
    """
    # Default parameters
    n_trials = 200
    sequence = None
    cue_validity = {'red': 0.75, 'green': 0.25}  # Probability of right movement
    
    if manipulation == 'biased_sequence':
        # Create a biased sequence: 80% red in first half, 80% green in second half
        first_half = np.random.choice(['red', 'green'], size=n_trials//2, p=[0.8, 0.2])
        second_half = np.random.choice(['red', 'green'], size=n_trials//2, p=[0.2, 0.8])
        sequence = np.concatenate([first_half, second_half])
        
    elif manipulation == 'switch_validity':
        # Standard sequence but switch validity halfway through
        def switch_experiment(model, n_trials):
            # First half
            first_half = np.random.choice(['red', 'green'], size=n_trials//2)
            for cue in first_half:
                model.simulate_trial(cue)
            
            # Switch validity
            model.cue_validity = {'red': 0.25, 'green': 0.75}
            
            # Second half
            second_half = np.random.choice(['red', 'green'], size=n_trials//2)
            for cue in second_half:
                model.simulate_trial(cue)
                
            return model
        
        model = PredictiveEyeMovementModel(cue_validity=cue_validity)
        model = switch_experiment(model, n_trials)
        df, summary = model.analyze_results(plot=True)
        return model, df, summary
        
    # Standard experiment
    model = PredictiveEyeMovementModel(cue_validity=cue_validity)
    model.simulate_experiment(n_trials=n_trials, sequence=sequence)
    df, summary = model.analyze_results(plot=True)
    
    return model, df, summary

# Run a standard simulation
model, results, summary = run_simulation()

# Analyze how anticipatory velocity depends on both cue and recent history
def analyze_history_dependency(results, window_size=5):
    """Analyze how recent history affects anticipatory behavior"""
    df = results.copy()
    
    # Create columns for recent history (% of rightward targets in last n trials)
    df['recent_right_ratio'] = 0
    
    for i in range(window_size, len(df)):
        recent_trials = df.iloc[i-window_size:i]
        right_count = sum(1 for dir in recent_trials['target_direction'] if dir == 'right')
        df.at[i, 'recent_right_ratio'] = right_count / window_size
    
    # Bin the recent history for visualization
    df['history_bin'] = pd.cut(df['recent_right_ratio'], 
                              bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                              labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    
    # Plot how anticipatory velocity depends on both cue and recent history
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='history_bin', y='anticipated_velocity', hue='cue', data=df.iloc[window_size:])
    plt.title(f'Anticipatory Velocity by Cue and Recent History (Last {window_size} Trials)')
    plt.xlabel('Recent History (% Rightward Targets)')
    plt.ylabel('Anticipatory Velocity')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return df

# Run this additional analysis
history_analysis = analyze_history_dependency(results)

# Function to demonstrate specific phenomena
def demonstrate_adaptation_to_statistics():
    """Show how the model adapts when statistical regularities change"""
    # Create a sequence with changing statistics
    n_trials = 400
    
    # Phase 1: Random cues but strong rightward bias regardless of cue
    cue_validity_phase1 = {'red': 0.9, 'green': 0.9}  # Both cues predict rightward
    
    # Phase 2: Standard validity where cues are predictive
    cue_validity_phase2 = {'red': 0.75, 'green': 0.25}
    
    # Create model with initial rightward bias for both cues
    model = PredictiveEyeMovementModel(cue_validity=cue_validity_phase1,
                                       initial_cue_weight=0.7)
    
    # Phase 1: Train with both cues predicting rightward
    sequence_phase1 = np.random.choice(['red', 'green'], size=n_trials//2)
    for cue in sequence_phase1:
        model.simulate_trial(cue)
        
    # Now switch to standard validity
    model.cue_validity = cue_validity_phase2
    
    # Phase 2: Train with standard validity
    sequence_phase2 = np.random.choice(['red', 'green'], size=n_trials//2)
    for cue in sequence_phase2:
        model.simulate_trial(cue)
    
    # Analyze results with a vertical line marking the phase change
    df = pd.DataFrame(model.trial_history)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['anticipated_velocity'], 'b-', alpha=0.6)
    plt.title('Adaptation to Changing Statistical Structure')
    plt.ylabel('Anticipatory Velocity')
    
    # Add cue color markers
    for i, (idx, row) in enumerate(df.iterrows()):
        color = 'red' if row['cue'] == 'red' else 'green'
        plt.scatter(i, row['anticipated_velocity'], color=color, s=20, alpha=0.7)
    
    # Add phase separator
    plt.axvline(x=n_trials//2, color='k', linestyle='--')
    plt.text(n_trials//4, 1.5, 'Phase 1: Both cues → Right', ha='center')
    plt.text(3*n_trials//4, 1.5, 'Phase 2: Standard validity', ha='center')
    
    plt.subplot(2, 1, 2)
    plt.plot(df['cue_weight'], 'r-', label='Cue Weight')
    plt.plot(df['history_weight'], 'b-', label='History Weight')
    plt.axvline(x=n_trials//2, color='k', linestyle='--')
    plt.xlabel('Trial')
    plt.ylabel('Weight')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, df

# Demonstrate adaptation to changing statistical structure
adaptation_model, adaptation_results = demonstrate_adaptation_to_statistics()
