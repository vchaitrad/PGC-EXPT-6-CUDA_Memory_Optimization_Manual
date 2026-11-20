\# EXPERIMENT 6: CUDA MEMORY OPTIMIZATION



\## AIM

To compare uncoalesced and coalesced memory access in matrix transpose, study shared memory padding, and execute naive matrix multiplication using CUDA.



\## HARDWARE

GPU: NVIDIA GeForce RTX 5060 Ti

GPU Memory: 16 GB

Driver Version: 617.42



\## SOFTWARE

CUDA Toolkit: 13.4

Compiler: NVCC V13.4.92

Platform: Windows



\## COMPILATION

nvcc -O2 -o exp6.exe exp6.cu



\## EXECUTION

exp6.exe



\## MATRIX TRANSPOSE RESULTS



| Run | Uncoalesced (ms) | Coalesced (ms) | Speedup |

|-----|------------------|----------------|---------|

| 1 | 0.797 | 0.414 | 1.93x |

| 2 | 0.798 | 0.414 | 1.93x |

| 3 | 0.798 | 0.412 | 1.94x |



\## PADDING COMPARISON



| Method | Coalesced Time | Bandwidth | Speedup |

|--------|----------------|-----------|---------|

| With Padding | 0.412 ms | 325.5 GB/s | 1.93x |

| Without Padding | 0.546 ms | 245.8 GB/s | 1.46x |



Padding improved the performance of the coalesced transpose in this experiment.



\## MATRIX MULTIPLICATION



Matrix Size: 1024 x 1024

Execution Time: 1.403 ms

Performance: 1530.8 GFLOPS

Verification: PASS

C\[0]\[0] = 1024.0



\## GRAPH

comparison\_graph.svg



\## SCREENSHOTS

Screenshots of three runs, GPU information, CUDA version and padding comparison are available in the screenshots folder.



\## CONCLUSION

Coalesced memory access improved transpose performance compared to uncoalesced access. Shared memory padding improved the measured performance by reducing bank conflicts. Matrix multiplication completed successfully and verification passed.

