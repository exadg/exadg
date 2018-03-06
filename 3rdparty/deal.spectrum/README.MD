```bash
./build/main "$(ls -v ../hilbert/build/file.*.vtk)"
```


./build/main ./build/input.bin
python scripts/creatematrix.py 0 3 4 3 build/input.bin
python scripts/parsematrix.py build/input.bin