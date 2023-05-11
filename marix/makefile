CXX := nvcc

SRC := $(wildcard *.cu)
OBJ := $(SRC:.cu=.o)
head := $(wildcard *.cuh)

mulmatrix : $(OBJ)
	$(CXX) $(OBJ) -o mulmatrix 

matrix.o : matrix.cu
	$(CXX) -c $< -o $@

main.o : main.cu $(head)
	$(CXX) -c $< -o $@

.PHONY : clean
clean:
	rm -f $(OBJ)