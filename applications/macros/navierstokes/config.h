#ifndef NAVIER_CONSTANTS
#define NAVIER_CONSTANTS

#define DIM_2 1
#define DIM_3 1

#define OP_FLOAT 0
#define OP_DOUBLE 1

#define MG_FLOAT 1
#define MG_DOUBLE 0

#define DEGREE_0 1
#define DEGREE_1 1
#define DEGREE_2 1
#define DEGREE_3 1
#define DEGREE_4 1
#define DEGREE_5 1
#define DEGREE_6 1
#define DEGREE_7 1
#define DEGREE_8 0
#define DEGREE_9 0
#define DEGREE_10 0
#define DEGREE_11 0
#define DEGREE_12 0
#define DEGREE_13 0
#define DEGREE_14 0
#define DEGREE_15 0

// incompressible Navier-Stokes: default is mixed-order (degree_p = degree_u -1)
// explicitly activate equal-order (degree_p = degree_u) here
#define EQUAL_ORDER 0

#endif
