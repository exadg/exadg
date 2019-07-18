Finite-element-based Navier-Stokes solvers
==========================================

This project provides efficient solvers for the Navier-Stokes equations based on
high-order discontinuous Galerkin finite element methods. By the use of efficient solution algorithms as well as solvers and preconditioners based on the matrix-free paradigm, this project aims at offering next generation's fluid solvers exploiting modern computer hardware and being prepared for the massively parallel era.

## Installation of software

As a prerequisite to get access to the code, a new user has to login at https://gitlab.lrz.de/ with the TUMOnline account **ab12xyz** so that the user can be added to the project.

### Structure of folders

Go to your *working_directory*

```bash
cd /working_directory/
```

**N.B.**: For students at LNM, the *scratch*-directory has to be used (as opposed to the home directory) as a folder for the subsequent installations

```bash
cd /scratch/students_name/
```
This directory is called *working_directory* in the following. 

For other users, the working directory might for example be

```bash
cd /home/username/
```

Add the following variable to your environment (in case you want to make the setting permanent, e.g. by inserting the following line into your *bashrc*-file)
```bash
export WORKING_DIRECTORY=/working_directory
```

We now create a folder called *workspace* in the *working_directory/* where we will later install the **navierstokes** code

```bash
mkdir workspace
```
Since we also have to install other software packages apart from the **navierstokes** code, we create another folder called *sw* (software) for third party software packages

```bash
mkdir sw
```
### navierstokes code (part 1)

Go to the *workspace*-folder in your working directory

```bash
cd /working_directory/workspace/
```

##### Fork navierstokes project

Fork from the supervisor's **navierstokes** project *git@gitlab.lrz.de:supervisor_id/navierstokes.git*, e.g.,

*git@gitlab.lrz.de:ga34jem/navierstokes.git* (Niklas) or 
*git@gitlab.lrz.de:ne96pad/navierstokes.git* (Martin). 

This has to be done on website https://gitlab.lrz.de/ (open the supervisor's **navierstokes** project and press the *Fork* button). As a result, a **navierstokes** project with the student's ID **ab12xyz** is created.

```bash
git clone https://gitlab.lrz.de/ab12xyz/navierstokes.git
cd navierstokes/
git remote add supervisor https://gitlab.lrz.de/supervisor_id/navierstokes.git
```

### Interlude - install other software packages

#### Trilinos code (optional)

For some functionalities in the **navierstokes** code (e.g., algebraic multigrid solver), **trilinos** is required. The default setting is to not install **trilinos** and installing this package is optional.

If you want to use **trilinos**, go to the *sw*-folder in your working directory

```bash
cd /working_directory/sw/
```
Download **trilinos** and run the following commands

```bash
wget https://github.com/trilinos/Trilinos/archive/trilinos-release-12-12-1.tar.gz
tar xf trilinos-release-12-12-1.tar.gz 
cd Trilinos-trilinos-release-12-12-1/

mkdir build
cd build/
```
Copy the script *config_trilinos.sh* from the folder *navierstokes/scripts/* to the current folder, e.g.,

```bash
cp /working_directory/workspace/navierstokes/scripts/config_trilinos.sh .
```
**N.B.**: To get these scripts, you first have to perform the first steps of the **navierstokes** installation described above, i.e., you have to fork and clone the **navierstokes** project.

Next, adapt the directory settings at the top of the script and run the script

```bash
bash ./config_trilinos.sh
```

For clusters @LNM: Load modules
```bash
module load mpi/openmpi-4.0.1
module load gcc/8
```
and adapt MPIDIR in *config_trilinos.sh*. Find out the path by
```bash
module show mpi/openmpi-4.0.1
```



Next, build the code

```bash
make -j2
make install
```

#### Metis code (optional)

For some functionalities in the **navierstokes** code (e.g., graph partitioning), **metis** is required. The default setting is to not install **metis** and installing this package is optional.

If you want to use **metis**, go to the *sw*-folder in your working directory

```bash
cd /working_directory/sw/
```

Download **metis** and run the following commands

```bash
git clone https://github.com/scibuilder/metis.git
cd metis
cmake .
make
```


#### deal.II code

The **navierstokes** code uses the **deal.II** library (https://www.dealii.org/), which is an open source finite element library based on the object-oriented C++ programming language.

Go to the *sw*-folder in your working directory

```bash
cd /working_directory/sw/
```

Clone the **deal.II** code from the gitlab project called **matrixfree**

```bash
git clone https://gitlab.lrz.de/ne96pad/matrixfree.git
```
Download **p4est**

```bash
wget http://p4est.github.io/release/p4est-2.0.tar.gz
```
and run the command

```bash
matrixfree/doc/external-libs/p4est-setup.sh p4est-2.0.tar.gz `pwd`
```
Create a folder *build*

```bash
mkdir build
cd build/
```
Copy the script *config_dealii.sh* from the folder *navierstokes/scripts/* to the current folder, e.g.,

```bash
cp /working_directory/workspace/navierstokes/scripts/config_dealii.sh .
```
**N.B.**: To get these scripts, you first have to perform the first steps of the **navierstokes** installation described above, i.e., you have to fork and clone the **navierstokes** project.

Next, adapt the directory settings at the top of the script and switch on trilinos/metis if desired (and adjust the folder if necessary)

```bash
...
-D DEAL_II_WITH_TRILINOS:BOOL="ON" \
-D DEAL_II_WITH_METIS:BOOL="ON" \
...
```
Run the config-script

```bash
bash ./config_dealii.sh
```

Build the **deal.II** code

```bash
make -j2
```

#### fftw code (optional)

Install **fftw** (Fast Fourier transformation) for evaluation of kinetic energy spectra:

Download **fftw** from homepage http://www.fftw.org/download.html and copy to folder *sw*

```bash
cd /working_directory/sw/
wget http://fftw.org/fftw-3.3.7.tar.gz
tar -xf fftw-3.3.7.tar.gz
cd fftw-3.3.7
./configure --enable-mpi --prefix=/working_directory/sw/fftw-3.3.7-install
make
make install
cd ../fftw-3.3.7-install/lib/
```
Copy the script *combine_fftw.sh* from the folder *navierstokes/scripts/* to the current folder, e.g.,

```bash
cp /working_directory/workspace/navierstokes/scripts/combine_fftw.sh .
```
**N.B.**: To get these scripts, you first have to perform the first steps of the **navierstokes** installation described above, i.e., you have to fork and clone the **navierstokes** project.

Run the script in order to combine the two libraries *libfftw3.a* and *libfftw3_mpi.a*

```bash
bash ./combine_fftw.sh
```
Add the following variable to your environment (in case you want to make the setting permanent, e.g. by inserting the following lines into your *bashrc*-file)

```bash
export FFTW_INC="$WORKING_DIRECTORY/sw/fftw-3.3.7-install/include"
export FFTW_LIB="$WORKING_DIRECTORY/sw/fftw-3.3.7-install/lib/combined"

```

### navierstokes code continued (part 2)

#### Link deal.II code and build the code

```bash
cd navierstokes/
```

Copy the script *config_navierstokes.sh* from the folder *navierstokes/scripts/* to the current folder, e.g.,

```bash
cp /working_directory/workspace/navierstokes/scripts/config_navierstokes.sh .
```

Deactivate the **fftw** related lines in *config_navierstokes.sh* if not needed, i.e., set

```bash
...
-D USE_DEAL_SPECTRUM=OFF \
...
```

and run the config-script 
```bash
bash ./config_navierstokes.sh
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
You can now run your first simulations by selecting a test case in one of the *my_application.cc* files (e.g., *incompressible_navier_stokes.cc*), setting the desired parameters in the *my_application_test_cases/my_test_case.h* header-file, and running

```bash
mpirun -np xxx ./my_application
```

#### Switching to debug-version

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

#### Working with git

Get recent updates of the supervisor's **navierstokes** code

```bash
git pull supervisor master
```
Commit changes and push:

Run *clang-format* for all files that have been changed, e.g.,

```bash
clang-format -i changed_file.cpp
clang-format -i new_file.h
```

Get an overview of what has been changed and add/commit. The following commands are used alternatingly

```bash
git status
git add changed_file.cpp new_file.h
git commit -m "a message describing what has been done/changed"
```

and finally push

```bash
git push
```

Start a merge-request on the website https://gitlab.lrz.de/:

Open your own **navierstokes** project, and press button *Merge Requests*. Select your own project as source and the supervisor's project as target.

#### Setup an eclipse project

Open **eclipse** and choose folder *workspace* as "workspace" in eclipse

1. File > New > Project > C/C++ > Makefile Project with Existing Code
  * fill in Project Name = navierstokes
  * Existing Code Location = /working_directory/workspace/navierstokes/
  * disable C, enable C++
  * choose Cross GCC
2. Project > Properties > C/C++ Build
  * use default build command or user specified build command, e.g., make -j4
  * fill in build directory (choose navierstokes/applications)
3. Project > Properties > C/C++ General > Code Analysis: disable 'syntax and semantic errors'
4. Project > Properties > C/C++ General > Formatter: lnm/styles/baci-eclipse-style
5. Project > Properties > C/C++ General > Paths and Symbols: use /working_directory/sw/matrixfree/include (for Assembly, GNU C, GNU C++)
6. Window > Preferences > General > Editors > Text Editors > Annotations > C/C++ Indexer Markers > uncheck all checkboxes > Apply > OK
