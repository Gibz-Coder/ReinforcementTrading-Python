#!/usr/bin/env python3
"""
Export trained model to ONNX format for MT5.
Uses legacy exporter for better compatibility.
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')


class SimplePolicyNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def export_to_onnx(model_path, output_dir="mt5_export"):
    print("=" * 50)
    print("EXPORTING MODEL TO ONNX")
    print("=" * 50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading: {model_path}")
    model = PPO.load(model_path)
    
    obs_shape = model.observation_space.shape
    n_actions = model.action_space.n
    input_size = int(np.prod(obs_shape))
    
    print(f"Input: {obs_shape} -> {input_size}")
    print(f"Output: {n_actions} actions")
    
    # Create network
    simple_net = SimplePolicyNet(input_size, [128, 128], n_actions)
    
    # Copy weights
    policy = model.policy
    state_dict = {
        'network.0.weight': policy.mlp_extractor.policy_net[0].weight.clone(),
        'network.0.bias': policy.mlp_extractor.policy_net[0].bias.clone(),
        'network.2.weight': policy.mlp_extractor.policy_net[2].weight.clone(),
        'network.2.bias': policy.mlp_extractor.policy_net[2].bias.clone(),
        'network.4.weight': policy.action_net.weight.clone(),
        'network.4.bias': policy.action_net.bias.clone(),
    }
    simple_net.load_state_dict(state_dict)
    simple_net.eval()
    
    # Test
    dummy = torch.randn(1, input_size, dtype=torch.float32)
    with torch.no_grad():
        out = simple_net(dummy)
    print(f"Test output shape: {out.shape}")
    
    # Export using legacy exporter
    onnx_path = os.path.join(output_dir, "trading_model.onnx")
    
    print(f"\nExporting with legacy exporter...")
    torch.onnx.export(
        simple_net,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['observations'],
        output_names=['action_logits'],
        dynamo=False  # Force legacy exporter
    )
    
    # Verify
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    file_size = os.path.getsize(onnx_path)
    opset = onnx_model.opset_import[0].version
    ops = sorted(set(n.op_type for n in onnx_model.graph.node))
    
    print(f"\nFile: {onnx_path}")
    print(f"Size: {file_size / 1024:.1f} KB")
    print(f"Opset: {opset}")
    print(f"Ops: {ops}")
    
    # Save config
    config = {
        'input_size': int(input_size),
        'window_size': int(obs_shape[0]),
        'num_features': int(obs_shape[1]),
        'num_actions': int(n_actions),
        'action_mapping': {'0': 'HOLD', '1': 'BUY', '2': 'SELL'}
    }
    
    with open(os.path.join(output_dir, "model_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nDone!")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, default='mt5_export')
    args = parser.parse_args()
    export_to_onnx(args.model, args.output)
