
HYPRE_DIR=${HOME}/hypre

mkdir -p build
cd build

CC=mpicc CXX=mpic++ cmake .. \
    -DHypre_INC_DIR=$HYPRE_DIR/include \
    -DHypre_LIB_DIR=$HYPRE_DIR/lib 
make -j3 

