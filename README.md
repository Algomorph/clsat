clsat
=====

OpenCL Summed Area Table implementation.
Currently tested only on NVIDIA cards (very likely to behave buggily on other platforms, needs further testing).

Employs the same strategy of using recursive filters / local memory to maximize throughput as in the following publication:
Nehab, D., Maximo, A., Lima, R. S., & Hoppe, H. (2011, December). GPU-efficient recursive filtering and summed-area tables. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 176). ACM.

