MAT_DIR = /home/lect0005/matrix
OBJ = main.o mmio.o io.o solver.o def.o help.o output.o errorcheck.o
SRC = $(OBJ:%.o=%.c?)
HDR = $(OBJ:%.o=%.h) 

C_FLAGS = ${FLAGS_FAST} -g 
LINKER = ${CC}

ifeq ($(dbg),1)
C_FLAGS = ${FLAGS_DEBUG} -DDEBUG
endif

default: openacc

openmp: C_FLAGS += ${FLAGS_OPENMP}
openmp: cg.exe

# To load the PGI compiler use: $ module switch intel pgi
openacc: CC = pgcc
openacc: C_FLAGS += -acc -Minfo=all -ta=nvidia,cc20,cuda6.5,keepptx,keepgpu,unroll,fma -Mlarge_arrays  -O4 -Msafeptr=all -pg
openacc: cg.exe


# To load the CUDA compiler use: $ module switch intel gcc; module load cuda
cuda: CC = icc
cuda: CUDA_CC = nvcc
cuda: LINKER = ${CUDA_CC}
cuda: C_FLAGS += -D__global__= -Wall -DCUDA -I/usr/local_rwth/sw/cuda/6.5.14/include/
ifeq ($(dbg),1)
cuda: CUDA_FLAGS += -O0 -DCUDA -DDEBUG ${FLAGS_DEBUG} -arch=sm_20 -G
else
cuda: CUDA_FLAGS += -ccbin icc -O3 -DCUDA -arch=sm_21 -DNO_TIMING -DNO_ERROR_CHECKS
# cuda: C_FLAGS += -g -march=native -mtune=native -funroll-loops
cuda: C_FLAGS += -opt-subscript-in-range -opt-prefetch -use-intel-optimized-headers -no-prec-div -O3 -prof-dir=pgo -xHost -fp-model fast=2 -restrict 
endif
cuda: cg.exe

cg.exe: ${OBJ}
	${LINKER} ${C_FLAGS} -o cg.exe ${OBJ} ${LINKER_FLAGS} 

%.o: %.c
	${CC} ${C_FLAGS} -c $<

%.o: %.cu
	${CUDA_CC} ${CUDA_FLAGS} -c $<


# For correctness you always should run all iterations to check convergence.
# For basic performance analysis it might be enough to run less iterations
# and compare the GFLOPS of the matrix vector product.
run: openacc
	CG_MAX_ITER=6000 ./cg.exe $(MAT_DIR)/G3_circuit.mtx

run_serena: openacc
	CG_MAX_ITER=6000 OMP_NUM_THREADS=12 ./cg.exe $(MAT_DIR)/Serena.mtx

run_debug: openacc
	CG_MAX_ITER=1 OMP_NUM_THREADS=1 OMP_PLACES=cores ./cg.exe debug_symm.mtx

clean:
	rm -f cg.exe
	rm -f *.o
