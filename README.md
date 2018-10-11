Finite-element-based Navier-Stokes solvers
==========================================

This is a collection of solvers for the Navier-Stokes equations based on
high-order discontinuous and continuous finite element method.

## Installation of software

As a prerequisite to get access to the code, a new user has to login at https://gitlab.lrz.de/ with the TUMOnline account **ab12xyz** so that the user can be added to the project.

**N.B.**: For students at LNM, the *scratch*-directory has to be used (as opposed to the home directory) as a folder for the subsequent installations. This directory is called *working_directory* in the following:

Go to *working_directory*, e.g.,

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
Copy script *compine.sh* to folder */working_directory/sw/fftw-3.3.7-install/lib/*

Run the script in order to combine the two libraries *libfftw3.a* and *libfftw3_mpi.a*

```bash
./combine.sh
```

### navierstokes code

The working directory is again */scratch/students_name/* for students at LNM. Create a folder named *workspace*

```bash
cd /working_directory/
mkdir workspace
cd workspace/
```
##### Fork navierstokes project

Fork from the supervisor's **navierstokes** project *git@gitlab.lrz.de:supervisor_id/navierstokes.git*, e.g.,

*git@gitlab.lrz.de:ga34jem/navierstokes.git* (Niklas) or 
*git@gitlab.lrz.de:ne96pad/navierstokes.git* (Martin). 

This has to be done on website https://gitlab.lrz.de/ (open the supervisor's **navierstokes** project and press the *Fork* button). As a result, a **navierstokes** project with the student's ID **ab12xyz** is created.

```bash
git clone git@gitlab.lrz.de:ab12xyz/navierstokes.git
git remote add supervisor git@gitlab.lrz.de:supervisor_id/navierstokes.git
```

##### Link deal.II code and build the code

```bash
cd navierstokes/
```
Run *cmake* (standard)

```bash
cmake -D DEAL_II_DIR=/working_directory/sw/build .
```
or use the following command if **fftw** is to be used (optional)

```bash
cmake -D FFTW_INC=/working_directory/sw/fftw-3.3.7-install/include -D FFTW_LIB=/working_directory/sw/fftw-3.3.7-install/lib/combined -D USE_DEAL_SPECTRUM=ON -D DEAL_II_DIR=/working_directory/sw/build .
```
In folder *navierstokes*, run the command

```bash
make release
```
and build **navierstokes** code

```bash
cd applications/
make -j2
```
You can now run your first simulations by selecting a test case in one of the *my_application.cc* files (e.g., *unsteady_navier_stokes.cc*), setting the desired parameters in the *my_application_test_cases/my_test_case.h* header-file, and running

```bash
mpirun -np xxx ./my_application
```

##### Switching to debug-version

To build the debug-version, run the following commands

```bash
cd ../
make debug
cd applications/
make -j2
```
and reactivate release-version after debugging via

```bash
cd ../
make release
cd applications/
make -j2
```
