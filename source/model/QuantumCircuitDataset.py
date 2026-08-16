#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantumCircuitDataset.py

This script contains the Quantum Circuit generation class.

__author__ = ""
__email__ = ""
"""

import cirq
import random
import numpy

from ..lib.rl4cirqopt.cirq_converter import import_from_cirq, export_to_cirq
from ..lib.rl4cirqopt.architecture import XmonArchitecture
from ..lib.rl4cirqopt.rules import (
    InvertCnot,
    CancelOperations,
    ExchangeCommutingOperations,
    ExchangePhasedXwithRotZ,
    # ExchangePhasedXwithControlledZ, # Not applicale with gate set
    CompressLocalOperations,
    TripleCnot,  # Created for expanding circuit
)


class CircuitDataset:
    def __init__(self, seed=1234, qubit_dim=12, num_gates=150):
        '''
        Class for generating random circuit and expanding random circuits via random transforms.

        Args:
            seed (int): The seed for random generation.
            qubit_dim: The number of qubits
            num_gates: The number of gates to add
        '''
        self.seed = seed
        random.seed(self.seed)
        numpy.random.seed(self.seed)

        self.qubit_dim = qubit_dim
        self.num_gates = num_gates
        self.architecture = XmonArchitecture()
        self.pruning_rules = [
            CancelOperations(),
            CompressLocalOperations(self.architecture)
        ]
        self.soft_rules = [
            InvertCnot(self.architecture),
            ExchangeCommutingOperations(),
            ExchangePhasedXwithRotZ(),
            TripleCnot(),
        ]
        return

    @staticmethod
    def to_rz_phx_cnot(circuit: cirq.Circuit) -> cirq.Circuit:
        """
        Decompose H, S, T, X, Y, Z into Rz and PhasedXPow.
        Convert CNOT into an explicit matrix gate representation.
        """
        cnot_matrix = numpy.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        cnot_gate = cirq.MatrixGate(cnot_matrix)

        translated = cirq.Circuit()
        for op in circuit.all_operations():
            gate = op.gate
            converted_ops = []

            if len(op.qubits) == 1:
                qubit = op.qubits[0]
                if isinstance(gate, cirq.HPowGate) and numpy.isclose(gate.exponent % 2, 1.0):
                    converted_ops.append(cirq.PhasedXPowGate(
                        exponent=0.5, phase_exponent=0.5).on(qubit))
                    converted_ops.append(cirq.rz(numpy.pi).on(qubit))
                elif isinstance(gate, cirq.XPowGate):
                    converted_ops.append(cirq.PhasedXPowGate(
                        exponent=gate.exponent, phase_exponent=0.0).on(qubit))
                elif isinstance(gate, cirq.YPowGate):
                    converted_ops.append(cirq.PhasedXPowGate(
                        exponent=gate.exponent, phase_exponent=0.5).on(qubit))
                elif isinstance(gate, cirq.ZPowGate):
                    converted_ops.append(
                        cirq.rz(numpy.pi * gate.exponent).on(qubit))
                else:
                    converted_ops.append(op)
            elif len(op.qubits) == 2 and op.gate == cirq.CNOT:
                converted_ops.append(cnot_gate.on(*op.qubits))
            else:
                converted_ops.append(op)

            translated.append(converted_ops)

        return translated

    def _generate_circuit(self):
        '''
        Generates a random circuit by sampling gates from:
        X, Y, Z, H, S, T, CNOT.

        Returns:
            cirq.Circuit: Random circuit translated to Rz/PhX/CNOT(Matrix) gate set.
        '''
        qubits = [cirq.LineQubit(i) for i in range(self.qubit_dim)]
        circuit = cirq.Circuit()

        one_qubit_gates = {
            "X": cirq.X,
            "Y": cirq.Y,
            "Z": cirq.Z,
            "H": cirq.H,
            "S": cirq.S,
            "T": cirq.T,
        }

        # p_cnot = 131.0 / 160.0
        p_cnot = 0.5
        local_gate_names = ["X", "Y", "Z", "H", "S", "T"]

        for _ in range(self.num_gates):
            if random.random() < p_cnot:
                q0 = random.randint(0, self.qubit_dim - 2)
                q1 = q0 + 1
                if random.random() < 0.5:
                    circuit.append(cirq.CNOT(qubits[q0], qubits[q1]))
                else:
                    circuit.append(cirq.CNOT(qubits[q1], qubits[q0]))
            else:
                q = random.choice(qubits)
                gate_name = random.choice(local_gate_names)
                circuit.append(one_qubit_gates[gate_name].on(q))

        return CircuitDataset.to_rz_phx_cnot(circuit)

    def _prune(self, rlcirq):
        for rule in self.pruning_rules:
            rlcirq = rule.apply_greedily(rlcirq)
        return rlcirq

    def _random_transformations(self, circuit):
        '''
        The procedure is as follows:
        Apply Pruning
        Apply 500 random transformations selected from soft rules
        Apply Pruning

        Args:
            circuit (cirq.Circuit): The randomly generated circuit
        Returns:
            cirq.Circuit: The expanded/randomly transformed circuit
        '''
        rlcirq = import_from_cirq(circuit)

        # Inital prune
        rlcirq = self._prune(rlcirq)

        # Random Transformation
        for i in range(500):
            select_rule = random.choice(self.soft_rules)
            possible_transforms = [t for t in select_rule.scan(rlcirq)]

            if possible_transforms:
                transformation = random.choice(possible_transforms)
                prev_depth = rlcirq.depth()
                new_rlcirq = transformation.perform()
                new_depth = new_rlcirq.depth()

                # Verify a increase of depth before preceeding
                if new_depth > prev_depth:
                    rlcirq = new_rlcirq

        # Final Prune
        rlcirq = self._prune(rlcirq)

        return export_to_cirq(rlcirq)

    def new_circuit(self):
        return self._random_transformations(self._generate_circuit())
