
MAT_DIR = /home/dk406646/swp-hpc/G3_circuit
OBJ = main.o mmio.o io.o solver.o def.o help.o output.o errorcheck.o 
SRC = $(OBJ:%.o=%.c)
HDR = $(OBJ:%.o=%.h) 

C_FLAGS = ${FLAGS_FAST} -restrict -g -opt-subscript-in-range -vec_report=2
LINKER = ${CC}

ifeq ($(dbg),1)
C_FLAGS = ${FLAGS_DEBUG} -DDEBUG -restrict -g
endif

default: openmp

openmp: C_FLAGS += ${FLAGS_OPENMP}
openmp: cg.exe

# To load the PGI compiler use: $ module switch intel pgi
openacc: CC = pgcc
openacc: C_FLAGS += -acc -Minfo=accel -ta=nvidia,cc20 -Mlarge_arrays
openacc: cg.exe


# To load the CUDA compiler use: $ module switch intel gcc; module load cuda
cuda: CC = gcc
cuda: CUDA_CC = nvcc
cuda: LINKER = ${CUDA_CC}
ifeq ($(dbg),1)
cuda: CUDA_FLAGS = -O0 -DCUDA -DDEBUG ${FLAGS_DEBUG} -G -arch=sm_20
else
cuda: CUDA_FLAGS = -O3 -DCUDA -arch=sm_20
cuda: C_FLAGS = -g
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
run: openmp
	OMP_PLACES=cores OMP_PROC_BIND=spread CG_MAX_ITER=6000 OMP_NUM_THREADS=12 ./cg.exe $(MAT_DIR)/G3_circuit.mtx

run_serena: openmp
	CG_MAX_ITER=20000 OMP_NUM_THREADS=64 ./cg.exe $(MAT_DIR)/Serena.mtx

run_debug: openmp
	CG_MAX_ITER=100 OMP_NUM_THREADS=6 OMP_PLACES=cores ./cg.exe debug.mtx

clean:
	rm -f cg.exe
	rm -f *.o
