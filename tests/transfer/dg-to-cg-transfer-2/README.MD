Solve two different problems `A_DG x = b_DG` and `A_CG x = b_CG` and compare the results 
and the number of iterations.
```
mkdir output
mpisplit -np 4 ./dg-to-cg-transfer-2
```
