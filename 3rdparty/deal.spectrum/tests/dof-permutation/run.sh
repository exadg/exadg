python scripts/morton-test.py 2 2 2 build/A build/B
mpirun -np 1 ./build/main build/A > build/temp
python scripts/morton-diff.py build/A_converted build/B

#python scripts/morton-test.py 2 8 5 build/A build/B
#mpirun -np 16 ./build/main build/A > build/temp
#python scripts/morton-diff.py build/A_converted build/B
