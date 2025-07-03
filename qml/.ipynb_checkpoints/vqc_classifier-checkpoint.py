import numpy as np
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from sklearn.metrics import accuracy_score

def build_vqc(X_train, y_train, X_test, y_test):
    backend = Aer.get_backend('aer_simulator')
    quantum_instance = QuantumInstance(backend)

    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    ansatz = RealAmplitudes(num_qubits=X_train.shape[1], reps=2)
    optimizer = COBYLA(maxiter=100)

    vqc = VQC(feature_map=feature_map,
              ansatz=ansatz,
              optimizer=optimizer,
              quantum_instance=quantum_instance)

    vqc.fit(X_train, y_train)
    predictions = vqc.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("Quantum Accuracy:", acc)

