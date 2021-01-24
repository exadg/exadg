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

Since we also have to install other software packages apart from the **ExaDG** code, we create another folder called *sw* (software) for third party software packages

```bash
mkdir sw
```


### ExaDG (first steps)

Go to the working directory

```bash
cd $WORKING_DIRECTORY
```

##### Forking ExaDG project

Fork from the supervisor's **ExaDG** project *git@gitlab.lrz.de:supervisor_id/exadg.git*, e.g.,

*git@gitlab.lrz.de:ga34jem/exadg.git* (Niklas) or 
*git@gitlab.lrz.de:ne96pad/exadg.git* (Martin). 

This has to be done on website https://gitlab.lrz.de/ (open the supervisor's **exadg** project and press the *Fork* button). As a result, an **ExaDG** project with the student's ID **ab12xyz** is created.

```bash
git clone https://gitlab.lrz.de/ab12xyz/exadg.git
cd exadg/
git remote add supervisor https://gitlab.lrz.de/supervisor_id/exadg.git
```


### Interlude - installing third party libraries

Go to the *sw*-folder in your working directory

```bash
cd $WORKING_DIRECTORY/sw/
```

#### Trilinos (optional)

For some functionalities in the **ExaDG** code (e.g., algebraic multigrid solver), **trilinos** is required. The default setting is to not install **trilinos** and installing this package is optional.

Download **trilinos** and run the following commands

```bash
wget https://github.com/trilinos/Trilinos/archive/trilinos-release-12-12-1.tar.gz
tar xf trilinos-release-12-12-1.tar.gz 
cd Trilinos-trilinos-release-12-12-1/

mkdir build
cd build/
```
Copy the script *config_trilinos.sh* from the folder *exadg/scripts/* to the current folder, e.g.,

```bash
cp $WORKING_DIRECTORY/exadg/scripts/config_trilinos.sh .
```
**N.B.**: To get these scripts, you first have to perform the first steps of the **ExaDG** installation described above, i.e., you have to fork and clone the **ExaDG** project.

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
make -j[N_CORES]
make install
```

#### Metis (optional)

For some functionalities in the **ExaDG** code (e.g., graph partitioning), **metis** is required. The default setting is to not install **metis** and installing this package is optional.

Download **metis** and run the following commands

```bash
git clone https://github.com/scibuilder/metis.git
cd metis
cmake .
make
```

#### deal.II

The **ExaDG** project uses the **deal.II** library (https://www.dealii.org/), which is an open source finite element library based on the object-oriented C++ programming language.

Clone the **deal.II** code

```bash
git clone https://github.com/dealii/dealii.git
```
Download **p4est**

```bash
wget http://p4est.github.io/release/p4est-2.0.tar.gz
```
and run the command

```bash
dealii/doc/external-libs/p4est-setup.sh p4est-2.0.tar.gz `pwd`
```
Create a *dealii-build* directory

```bash
mkdir dealii-build
cd dealii-build/
```
Copy the script *config_dealii.sh* from the folder *exadg/scripts/* to the current folder, e.g.,

```bash
cp $WORKING_DIRECTORY/exadg/scripts/config_dealii.sh .
```
**N.B.**: To get these scripts, you first have to perform the first steps of the **ExaDG** installation described above, i.e., you have to fork and clone the **ExaDG** project.

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
make -j[N_CORES]
```

#### fftw (optional)

Install **fftw** (Fast Fourier transformation) for evaluation of kinetic energy spectra:

Download **fftw** from homepage http://www.fftw.org/download.html and copy to folder *sw*

```bash
wget http://fftw.org/fftw-3.3.7.tar.gz
tar -xf fftw-3.3.7.tar.gz
cd fftw-3.3.7
./configure --enable-mpi --prefix=$WORKING_DIRECTORY/sw/fftw-3.3.7-install
make
make install
cd ../fftw-3.3.7-install/lib/
```
Copy the script *combine_fftw.sh* from the folder *exadg/scripts/* to the current folder, e.g.,

```bash
cp $WORKING_DIRECTORY/exadg/scripts/combine_fftw.sh .
```

**N.B.**: To get these scripts, you first have to perform the first steps of the **ExaDG** installation described above,
i.e., you have to fork and clone the **ExaDG** project.

Run the script in order to combine the two libraries *libfftw3.a* and *libfftw3_mpi.a*

```bash
bash ./combine_fftw.sh
```

### Likwid (optional)

Download likwid release 4.3.3 from github to folder *sw*

```bash
wget https://github.com/RRZE-HPC/likwid/archive/likwid-4.3.3.tar.gz
```

Unzip the file with tar

```bash
tar -xf likwid-4.3.3.tar.gz
```

Change into the likwid directory

```bash
cd likwid-likwid-4.3.3
```

Open the likwid config file

```bash
vi config.mk
```

Select the install directory, for example

```bash
PREFIX = $(WORKING_DIRECTORY)/sw/likwid-install
```

Build likwid and install it into the selected folder (*sudo* required to install the access daemon with proper
permissions)

```bash
make
sudo make install
```

Set the likwid install directory in *config_exadg.sh* (see next step)

### Completing ExaDG installation (continued)

#### Linking deal.II code and building the code

```bash
cd $WORKING_DIRECTORY/exadg/
mkdir build
cd build/
```

Copy the script *config_exadg.sh* from the *exadg/scripts/* directory to the *exadg/build/* directory, e.g.,

```bash
cp $WORKING_DIRECTORY/exadg/scripts/config_exadg.sh .
```

Deactivate the **fftw** related lines in *config_exadg.sh* if not needed, i.e., set

```bash
...
-D USE_DEAL_SPECTRUM=OFF \
...
```

and run the config-script 
```bash
bash ./config_exadg.sh
```

Next, run the command

```bash
make release
```
and build the code

```bash
make -j[N_CORES]
```

#### Running simulations in **ExaDG**

To run your first simulations, select a solver, e.g., *incompressible_navier_stokes*, and of the flow examples for this solver in the *exadg/solvers/incompressible_navier_stokes/applications/* directory, where you can set and modify the parameters of the considered flow problem.

```bash
cd solvers/incompressible_navier_stokes/
mpirun -np [N_CORES] ./solver $WORKING_DIRECTORY/exadg/solvers/incompressible_navier_stokes/applications/my_application/input.json
```

#### Debugging

To build the debug-version, run the following commands

```bash
cd $WORKING_DIRECTORY/exadg/build/
make debug
make -j[N_CORES]
```
Debug code with **gdb**
```bash
cd solvers/incompressible_navier_stokes/
gdb --args ./solver path_to_application/input.json
```

Don't forget to reactivate release-version after debugging via

```bash
cd $WORKING_DIRECTORY/exadg/build/
make release
make -j[N_CORES]
```

#### Working with git

Get recent updates of the supervisor's project

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

Open your own project, and press button *Merge Requests*. Select your own project as source and the supervisor's project as target.

#### Setting up an eclipse project

Start **eclipse** and choose the *working_directory/* as "workspace" in eclipse

1. File > New > Project > C/C++ > Makefile Project with Existing Code
  * fill in Project Name = exadg
  * Existing Code Location = /working_directory/exadg/
  * disable C, enable C++
  * choose Cross GCC
2. Project > Properties > C/C++ Build
  * use default build command or user specified build command, e.g., make -j4
  * fill in build directory (choose *exadg/build/* directory)
3. Project > Properties > C/C++ General > Code Analysis: disable 'syntax and semantic errors'
4. Project > Properties > C/C++ General > Paths and Symbols: use /working_directory/dependencies/dealii/include (for Assembly, GNU C, GNU C++)
5. Window > Preferences > General > Editors > Text Editors > Annotations > C/C++ Indexer Markers > uncheck all checkboxes > Apply > OK
