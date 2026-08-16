#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantCircOptEnv.py

This script contains the reinforcement learning environment class.

__author__ = ""
__email__ = ""
"""

import cirq
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .QuantumCircuitDataset import CircuitDataset

from ..lib.rl4cirqopt import cirq_converter
from ..lib.rl4cirqopt.circuit import Circuit
from ..lib.rl4cirqopt.rules import (
    InvertCnot,
    ExchangeCommutingOperations,
    ExchangePhasedXwithRotZ,
    CancelOperations,
    CompressLocalOperations,
)
from ..lib.rl4cirqopt.architecture import XmonArchitecture


class QuantCircOptEnv(gym.Env):
    """
    Reinforcement learning environment for quantum circuit optimization.
    """

    def __init__(self, qubit_dim=12, depth_dim=1000, num_gates=150, seed=1234):
        '''
        Constructor inputs

        Args:
            qubit_dim (int): The number of qubits, and qubit dimension number of observation/policy output
            depth_dim (int): The depth dimension number of observation/policy output
            num_gates (int): The number of gates in random circuit to apply random transformations and pruning
        '''
        super().__init__()
        # Required component of gymnasium env
        # Maximum circuit dimension during training or testing
        # Fully convolutional model means dimension is adjustable

        self.qubit_dim = qubit_dim
        self.depth_dim = depth_dim
        self.num_gates = num_gates
        self.arch = XmonArchitecture()

        self.seed = seed
        self.dataset = CircuitDataset(
            self.seed, self.qubit_dim, self.num_gates)

        self._transform_grid = None
        self._mask_grid = None

        self.current_step = 0
        self.max_steps = 600
        self.episode_reward = 0.0

        # Define rules for the optimizer
        self.rules = [
            InvertCnot(self.arch),
            ExchangeCommutingOperations(),
            ExchangePhasedXwithRotZ(),
            # ExchangePhasedXwithControlledZ(XmonArchitecture) # Gate set is PhX, Rz, CNOT
        ]

        # SB3 CNNs require Channel-First: (Channels, Height, Width)
        # Low=0, High=8 (based on _encode_circuit_to_image mapping)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0,
                    high=7,
                    shape=(1, qubit_dim, depth_dim),
                    dtype=np.uint8,
                ),
                "mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(self.rules), qubit_dim, depth_dim),
                    dtype=np.uint8,
                ),
            }
        )

        # Action space: flattened index of the (qubit, depth) grid
        # This allows the model to output a policy "image" with rule channels
        self.action_space = spaces.Discrete(
            len(self.rules) * qubit_dim * depth_dim)

        self.CNOT_MATRIX = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 0, 1],
                                     [0, 0, 1, 0]], dtype=np.complex128)

        self.reset()

    def _optimization_stats(self, verbose=False):
        '''
        Returns the gates and depth of the baseline and optimized circuit

        Args:
            verbose (bool): Whether to print the stats
        Returns:
            list: [depth_baseline, gates_baseline, depth_optimized, gates_optimized]
        '''
        gates_optimized = 0
        depth_optimized = len(cirq_converter.export_to_cirq(self.circuit))
        for moment in cirq_converter.export_to_cirq(self.circuit):
            for gate in moment:
                gates_optimized += 1

        gates_baseline = 0
        depth_baseline = len(cirq_converter.export_to_cirq(self.baseline_circ))
        for moment in cirq_converter.export_to_cirq(self.baseline_circ):
            for gate in moment:
                gates_baseline += 1

        if verbose:
            print("Optimized Gate #: ", gates_optimized)
            print("Optimized Depth: ", depth_optimized)
            print("Original Gate #: ", gates_baseline)
            print("Original Depth: ", depth_baseline)

        return [depth_baseline, gates_baseline, depth_optimized, gates_optimized]

    def _transformation_location(self, trans):
        """
        Calculates the specific qubit and moment for a transformation context.
        """
        context_before_circ = trans._attention_circ._context._before
        focus = trans._attention_circ._focus

        current_transform_qubits = set()
        for operation in focus:
            # Handle list of qubits or single qubit objects
            qs = (
                operation._qubits
                if isinstance(operation._qubits, (list, tuple))
                else [operation._qubits]
            )
            current_transform_qubits.update(qs)

        if context_before_circ:
            last_op_qubits = context_before_circ[-1]._qubits
            # logic to determine if the transform shares a moment or starts a new one
            if not set(last_op_qubits).isdisjoint(current_transform_qubits):
                moment = len(cirq_converter.export_to_cirq(
                    context_before_circ))
            else:
                moment = len(cirq_converter.export_to_cirq(
                    context_before_circ)) - 1
        else:
            moment = 0

        return current_transform_qubits, moment

    def _rebuild_transform_cache(self):
        """
        Build transformation lookup grid and corresponding action mask for current circuit.
        """
        transform_grid = []
        for _ in range(len(self.rules)):
            rule_layer = []
            for _ in range(self.qubit_dim):
                depth_row = []
                for _ in range(self.depth_dim):
                    depth_row.append(None)
                rule_layer.append(depth_row)
            transform_grid.append(rule_layer)

        action_mask = np.zeros(
            (len(self.rules), self.qubit_dim, self.depth_dim),
            dtype=np.uint8,
        )

        for rule_idx, rule in enumerate(self.rules):
            for trans in rule.scan(self.circuit):
                qubit_loc, moment = self._transformation_location(trans)
                if moment >= self.depth_dim:
                    raise ValueError(
                        "Moment > Depth. Increase the depth in the constructor.")

                q_idx = min(qubit_loc)
                transform_grid[rule_idx][q_idx][moment] = trans
                action_mask[rule_idx, q_idx, moment] = 1

        self._transform_grid = transform_grid
        self._mask_grid = action_mask

    def _encode_circuit_to_image(self, circuit: Circuit):
        """
        Converts a Circuit into a multi-value image grid.
        0 - None
        1 - PhX
        2 - Rz Misc
        3 - Rz pi/2
        4 - Rz pi
        5 - Rz 3pi/2
        6 - CNOT control
        7 - CNOT target
        """
        cirq_circ = cirq_converter.export_to_cirq(circuit)
        image = np.zeros((self.qubit_dim, self.depth_dim), dtype=np.uint8)

        qubit_list = sorted(cirq_circ.all_qubits())

        for t, moment in enumerate(cirq_circ):
            if t >= self.depth_dim:
                raise RuntimeError("Image is too small for circuit encoding")
            for op in moment.operations:
                qubit_indices = [qubit_list.index(q) for q in op.qubits]
                gate = op.gate

                # Mapping logic as per your specs
                for q in qubit_indices:
                    if q >= self.qubit_dim:
                        continue
                    if isinstance(gate, cirq.PhasedXPowGate):
                        image[q, t] = 1
                    # single-qubit Rz-like rotation
                    elif cirq.unitary(gate).shape == (2, 2):
                        # S gate    - Rz(pi/2)
                        # Z gate    - Rz(pi)
                        # S^-1 gate - Rz(3pi/2)
                        # rest is misc.
                        if gate == cirq.S:
                            image[q, t] = 3
                        elif gate == cirq.Z:
                            image[q, t] = 4
                        elif gate == cirq.S**-1:
                            image[q, t] = 5
                        else:
                            image[q, t] = 2
                    elif cirq.unitary(gate).shape == (4, 4) and np.allclose(cirq.unitary(gate), self.CNOT_MATRIX):
                        # Control/Target distinction
                        # qubit index 0 is the control qubit
                        image[qubit_indices[0], t] = 6
                        # qubit index 1 is the target qubit
                        image[qubit_indices[1], t] = 7
        return image

    def step(self, action):
        action = int(np.asarray(action).item())
        # Translate flat action index back to (q, t)
        plane = self.qubit_dim * self.depth_dim
        rule_idx = action // plane
        rem = action % plane
        q_idx = rem // self.depth_dim
        t_idx = rem % self.depth_dim

        # Find and apply the transformation corresponding to this location
        trans = self._transform_grid[rule_idx][q_idx][t_idx]
        if trans is not None:
            self.circuit = trans.perform()

        # Apply all hard rules
        for trans in [CancelOperations(), CompressLocalOperations(self.arch)]:
            self.circuit = trans.apply_greedily(self.circuit)

        # compute reward: q_prev - q_current (positive reward for reduction)
        cirq_circuit = cirq_converter.export_to_cirq(self.circuit)
        depth = len(cirq_circuit)
        gates = len(list(cirq_circuit.all_operations()))

        q_curr = depth + 0.2 * gates
        reward = self.q_prev - q_curr
        self.q_prev = q_curr

        self.episode_reward += float(reward)

        self._rebuild_transform_cache()
        image = self._encode_circuit_to_image(self.circuit)[np.newaxis, :, :]
        mask_grid = self._mask_grid
        observation = {"obs": image, "mask": mask_grid}

        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.max_steps  # auto-truncate at max steps

        info = {"mask": mask_grid}

        if truncated or terminated:
            # final episode compute the circuit stats
            optimization_stats = self._optimization_stats(verbose=False)

            # Keep custom episode stats under a separate key so Monitor can
            # still populate info["episode"] with r/l/t.
            info["opt_episode"] = {
                "baseline_depth": optimization_stats[0],
                "baseline_gates": optimization_stats[1],
                "optimized_depth": optimization_stats[2],
                "optimized_gates": optimization_stats[3],
                "total_reward": self.episode_reward,
            }
            self.current_step = 0
            self.episode_reward = 0.0

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # random transformations
        circuit = self.dataset.new_circuit()

        self.baseline_circ = cirq_converter.import_from_cirq(circuit)
        self.circuit = self.baseline_circ

        cirq_circuit = cirq_converter.export_to_cirq(self.circuit)
        self.q_prev = len(cirq_circuit) + 0.2 * \
            len(list(cirq_circuit.all_operations()))
        self.episode_reward = 0.0

        self._rebuild_transform_cache()
        image = self._encode_circuit_to_image(self.circuit)[np.newaxis, :, :]
        mask = self._mask_grid
        observation = {"obs": image, "mask": mask}

        return observation, {}
