import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

class DQNModel(nn.Module):
    """PyTorch implementation of the DQN model used in tetris-ai-master"""
    
    def __init__(self, input_dim, hidden_layers=[32, 32], activations=['relu', 'relu', 'linear']):
        super(DQNModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        
        # Output layer (single value)
        self.layers.append(nn.Linear(hidden_layers[-1], 1))
        
        # Store activations
        self.activations = []
        self.activation_names = []  # Store names for comparison
        for act in activations:
            if act == 'relu':
                self.activations.append(nn.ReLU())
                self.activation_names.append('relu')
            elif act == 'tanh':
                self.activations.append(nn.Tanh())
                self.activation_names.append('tanh')
            elif act == 'sigmoid':
                self.activations.append(nn.Sigmoid())
                self.activation_names.append('sigmoid')
            else:  # linear or unknown
                self.activations.append(nn.Identity())  # Use Identity instead of lambda
                self.activation_names.append('linear')
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation (except for the last layer if it's linear)
            if i < len(self.activations):
                # Check if it's the last layer with linear activation
                if not (i == len(self.layers) - 1 and self.activation_names[i] == 'linear'):
                    x = self.activations[i](x)
        return x


class PyTorchDQNAgent:
    """PyTorch implementation of the DQN agent in tetris-ai-master"""
    
    def __init__(self, state_size, mem_size=10000, discount=0.95,
                epsilon=1, epsilon_min=0, epsilon_stop_episode=0,
                n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                lr=0.001, replay_start_size=None, model_file=None):
        
        self.state_size = state_size
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        
        # Exploration parameters
        if epsilon_stop_episode > 0:
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        else:  # no random exploration
            self.epsilon = 0
        
        # Network parameters
        self.n_neurons = n_neurons
        self.activations = activations
        self.lr = lr
        
        if not replay_start_size:
            replay_start_size = mem_size // 2
        self.replay_start_size = replay_start_size
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load existing model or create new one
        if model_file is not None and os.path.exists(model_file):
            self.model = self._load_model(model_file)
        else:
            self.model = self._build_model()
        
        # Move model to device (ensure it's a model, not state_dict)
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        else:
            # If it's a state_dict, create model and load it
            model = self._build_model()
            model.load_state_dict(self.model)
            self.model = model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def _build_model(self):
        """Build a new PyTorch model"""
        return DQNModel(self.state_size, self.n_neurons, self.activations)
    
    def _load_model(self, model_file):
        """Load model from file (either Keras or PyTorch)"""
        if model_file.endswith('.keras') or model_file.endswith('.h5'):
            # Convert Keras model to PyTorch
            print(f"Converting Keras model {model_file} to PyTorch...")
            
            try:
                import tensorflow as tf
                
                # Load Keras model
                keras_model = tf.keras.models.load_model(model_file)
                print(f"Loaded Keras model with {len(keras_model.layers)} layers")
                
                # Print Keras model architecture for debugging
                print("Keras model architecture:")
                for i, layer in enumerate(keras_model.layers):
                    weights = layer.get_weights()
                    if weights:
                        print(f"  Layer {i}: {layer.name}, weights shape: {[w.shape for w in weights]}")
                    else:
                        print(f"  Layer {i}: {layer.name}, no weights")
                
                # Build PyTorch model
                model = self._build_model()
                print(f"Built PyTorch model with {len(list(model.layers))} layers")
                
                # Print PyTorch model architecture for debugging
                print("PyTorch model architecture:")
                for i, layer in enumerate(model.layers):
                    print(f"  Layer {i}: weight shape {layer.weight.shape}, bias shape {layer.bias.shape}")
                
                # Transfer weights layer by layer
                keras_layer_idx = 0
                for pytorch_layer_idx, pytorch_layer in enumerate(model.layers):
                    # Find next Keras layer with weights
                    while keras_layer_idx < len(keras_model.layers):
                        keras_weights = keras_model.layers[keras_layer_idx].get_weights()
                        if len(keras_weights) >= 2:  # Has both weights and bias
                            break
                        keras_layer_idx += 1
                    
                    if keras_layer_idx >= len(keras_model.layers):
                        print(f"Warning: No more Keras layers with weights for PyTorch layer {pytorch_layer_idx}")
                        break
                    
                    # Get Keras weights (weight matrix and bias vector)
                    keras_weight = keras_weights[0]  # Shape: (input_dim, output_dim)
                    keras_bias = keras_weights[1]    # Shape: (output_dim,)
                    
                    print(f"Transferring layer {pytorch_layer_idx}:")
                    print(f"  Keras weight shape: {keras_weight.shape}")
                    print(f"  Keras bias shape: {keras_bias.shape}")
                    print(f"  PyTorch weight shape: {pytorch_layer.weight.shape}")
                    print(f"  PyTorch bias shape: {pytorch_layer.bias.shape}")
                    
                    # Convert to PyTorch tensors and transpose weight matrix
                    # Keras: (input_dim, output_dim) -> PyTorch: (output_dim, input_dim)
                    pytorch_weight = torch.FloatTensor(keras_weight.T)  # Transpose!
                    pytorch_bias = torch.FloatTensor(keras_bias)
                    
                    # Verify shapes match
                    if pytorch_weight.shape != pytorch_layer.weight.shape:
                        raise ValueError(f"Weight shape mismatch: Keras {keras_weight.shape} -> PyTorch {pytorch_weight.shape} vs expected {pytorch_layer.weight.shape}")
                    if pytorch_bias.shape != pytorch_layer.bias.shape:
                        raise ValueError(f"Bias shape mismatch: Keras {keras_bias.shape} -> PyTorch {pytorch_bias.shape} vs expected {pytorch_layer.bias.shape}")
                    
                    # Assign weights to PyTorch model
                    pytorch_layer.weight.data = pytorch_weight
                    pytorch_layer.bias.data = pytorch_bias
                    
                    print(f"  ✓ Successfully transferred layer {pytorch_layer_idx}")
                    keras_layer_idx += 1
                
                print("✓ Keras to PyTorch conversion completed successfully!")
                
                # Test the converted model with a sample input
                print("Testing converted model...")
                test_input = torch.randn(1, self.state_size)
                with torch.no_grad():
                    output = model(test_input)
                    print(f"Sample input shape: {test_input.shape}")
                    print(f"Sample output: {output.item():.6f}")
                    print("✓ Model forward pass successful!")
                
                return model
                
            except Exception as e:
                print(f"❌ Error converting Keras model: {e}")
                import traceback
                traceback.print_exc()
                print("Creating a new random PyTorch model instead.")
                return self._build_model()
        
        else:
            # Load PyTorch model
            try:
                print(f"Loading PyTorch model from {model_file}")
                # Try to load as a full model first
                try:
                    model = torch.load(model_file, weights_only=False, map_location='cpu')
                    print("✓ Loaded as full PyTorch model")
                    return model
                except Exception:
                    # If that fails, try loading as state_dict
                    model = self._build_model()
                    state_dict = torch.load(model_file, weights_only=False, map_location='cpu')
                    model.load_state_dict(state_dict)
                    print("✓ Loaded as PyTorch state_dict")
                    return model
            except Exception as e:
                print(f"❌ Error loading PyTorch model: {e}")
                import traceback
                traceback.print_exc()
                print("Creating a new random model instead.")
                return self._build_model()
    
    def add_to_memory(self, current_state, next_state, reward, done):
        """Add a transition to the replay memory"""
        self.memory.append((current_state, next_state, reward, done))
    
    def random_value(self):
        """Generate random score for exploration"""
        return random.random()
    
    def predict_value(self, state):
        """Predict value for a state using the model"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            return self.model(state_tensor).item()
    
    def act(self, state):
        """Return expected score for a state (with epsilon-greedy)"""
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)
    
    def best_state(self, states):
        """Return the best state from a collection of states"""
        max_value = None
        best_state = None
        
        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            for state in states:
                state_array = np.reshape(state, [1, self.state_size])
                value = self.predict_value(state_array)
                if max_value is None or value > max_value:
                    max_value = value
                    best_state = state
        
        return best_state
    
    def train(self, batch_size=32, epochs=3):
        """Train the agent using experience replay"""
        if batch_size > self.mem_size:
            print('WARNING: batch size is bigger than mem_size. The agent will not be trained.')
            return
        
        n = len(self.memory)
        if n >= self.replay_start_size and n >= batch_size:
            # Sample random batch from memory
            batch = random.sample(self.memory, batch_size)
            
            # Get next states for Q-learning targets
            next_states = np.array([x[1] for x in batch])
            next_states_tensor = torch.FloatTensor(next_states).to(self.device)
            
            with torch.no_grad():
                next_qs = self.model(next_states_tensor).cpu().numpy()
            
            # Prepare training data
            states = np.array([x[0] for x in batch])
            states_tensor = torch.FloatTensor(states).to(self.device)
            
            # Build target values
            targets = []
            for i, (_, _, reward, done) in enumerate(batch):
                if not done:
                    # Q-learning target
                    new_q = reward + self.discount * next_qs[i][0]
                else:
                    new_q = reward
                targets.append([new_q])
            
            targets_tensor = torch.FloatTensor(targets).to(self.device)
            
            # Train for multiple epochs
            for _ in range(epochs):
                # Forward pass
                predictions = self.model(states_tensor)
                
                # Compute loss and backward pass
                loss = self.criterion(predictions, targets_tensor)
                
                # Zero gradients, backward pass, optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
    
    def save_model(self, name):
        """Save the PyTorch model"""
        # Save only the state_dict instead of the entire model for better compatibility
        torch.save(self.model.state_dict(), name)


# For backward compatibility with the existing code
class DQNAgent(PyTorchDQNAgent):
    """Compatibility wrapper for the original Keras-based DQNAgent"""
    
    def __init__(self, state_size, mem_size=10000, discount=0.95,
                epsilon=1, epsilon_min=0, epsilon_stop_episode=0,
                n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                loss='mse', optimizer='adam', replay_start_size=None, modelFile=None):
        
        # Map optimizer string to learning rate
        lr = 0.001  # default for Adam
        if optimizer == 'rmsprop':
            lr = 0.0001
        
        super().__init__(state_size, mem_size, discount, epsilon, epsilon_min, 
                         epsilon_stop_episode, n_neurons, activations, lr, 
                         replay_start_size, modelFile) 