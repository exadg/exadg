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

Create a folder *sw* (software) where to install the deal.II code

```bash
mkdir sw
cd sw/
```
Clone the **deal.II** code from the gitlab project called **matrixfree**

```bash
git clone https://gitlab.lrz.de/ne96pad/matrixfree.git
```
Download **p4est**

```bash
wget http://p4est.github.io/release/p4est-1.1.tar.gz
```
and run the command

```bash
matrixfree/doc/external-libs/p4est-setup.sh p4est-1.1.tar.gz `pwd`
```
Create a folder *build*

```bash
mkdir build
cd build/
```
Copy the file *config.mpi* to the build-folder and adjust folders in *config.mpi* to the folders on the local machine (write **matrixfree** isntead of **deal.II** in the last line)

```bash
./config.mpi
make -j2
```

### fftw code (optional)

Install **fftw** (Fast Fourier transformation) for evaluation of kinetic energy spectra

download **fftw** from homepage http://www.fftw.org/download.html and copy to folder *sw*

```bash
cd fftw-3.3.7
./configure --enable-mpi --prefix=/scratch/students_name/sw/fftw-3.3.7-install
make
make install
cd ../fftw-3.3.7-install/lib/
```
Copy script *compine.sh* to folder */scratch/students_name/sw/fftw-3.3.7-install/lib/*

Run the script in order to combine the two libraries *libfftw3.a* and *libfftw3_mpi.a*

```bash
./combine.sh
```
