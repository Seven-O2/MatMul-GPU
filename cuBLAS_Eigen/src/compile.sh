nvcc kernel.cu -lcublas --expt-relaxed-constexpr -O2 -Xcompiler -fopenmp -Xcompiler -mavx -o kernel
