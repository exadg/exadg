Finite-element-based Navier-Stokes solvers
==========================================

This is a collection of solvers for the Navier-Stokes equations based on
high-order discontinuous and continuous finite element method.

## Installation of software

As a prerequisite to get access to the code, a new user has to login at https://gitlab.lrz.de/ with the TUMOnline account **ab12xyz** so that the user can be added to the project.

**N.B.**: For students at LNM, the scratch-directory has to be used (as opposed to the home directory) as a folder for the subsequent installations:

```bash
cd /scratch/students_name/
```

### deal.II code

Create a folder where to install the deal.II code

```bash
mkdir sw
cd sw/
```
Clone the deal.II code from the gitlab project called **matrixfree**

```bash
git clone https://gitlab.lrz.de/ne96pad/matrixfree.git
```