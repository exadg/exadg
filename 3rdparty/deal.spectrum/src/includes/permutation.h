/*
 * <DEAL.SPECTRUM>/includes/permutation.h
 *
 *  Created on: Mar 02, 2018
 *      Author: muench
 */

#ifndef DEAL_SPECTRUM_PERMUTATION
#define DEAL_SPECTRUM_PERMUTATION

// include std
#include <map>
#include <math.h> 
#include <mpi.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

// include DEAL.SPECTRUM modules
#include "./setup.h"

namespace dealspectrum{

// data structures needed for the following comparator for index sorting
int *array_list;    // position in local array ...
int *array_proc;    // element resides on process...
int __rank;         // rank of this process

/**
 * Special comparator for sorting of indices such that:
 *      - process numbers of elements (in array_proc) are in ascending order
 *      - elements residing on this process (identified via __rank) come first
 *      - elements are in ascending order according array__list for each process
 * 
 * @param a     compare a ...
 * @param b     ... to b
 * @return      order
 */
int cmp(const void *a, const void *b){
    int ia = *(int *)a;
    int ib = *(int *)b;
    return
    array_proc[ia] == array_proc[ib] ? 
        // same process
        array_list[ia] < array_list[ib] ? -1 : array_list[ia] > array_list[ib] :
        // different processes 
        array_proc[ia] == __rank ? -1 :
        array_proc[ib] == __rank ? +1 :
        array_proc[ia] < array_proc[ib] ? -1 : array_proc[ia] > array_proc[ib];
}

/**
 * Class for permuting dofs from cellwise-sfc-order to lexicographical order
 * such that processes own complete rows (2D) or complete planes (3D) as 
 * specified by FFTW. To be able to process the resulting array by FFTW, we 
 * on the fly insert padding.
 * 
 * This class expects an input array as described in the interpolation class and
 * produces an array of the format which comply the requirements described in 
 * the FFTW wrapper class. 
 */
class Permutator {
    // reference to DEAL.SPECTRUM setup
    Setup& s;
    // is initialized?
    bool initialized;
    
public:
    /**
     * Constructor
     * @param s DEAL.SPECTRUM setup
     */
    Permutator(Setup& s) : s(s), initialized(false){ }

    /**
     * Initialize all data structures needed for communication
     * 
     * @param MAP       reference to bijection function
     * @param FFT       reference to fftw-wrapper
     */
    template <typename MAPPING, typename FFTW>
    void init(MAPPING& MAP, FFTW & FFT){
        
        // check if already initialized
        if(this->initialized) return;
        this->initialized = true;

        // help variables...
            dim    = s.dim;
        int rank   = s.rank;
        int points = s.points_dst;
        int n      = s.cells;
    
        int start, end, start_, end_;
        // ... my range of the space filling curve
        MAP.getLocalRange(start, end);
        // ... my rows for FFT
        FFT.getLocalRange(start_, end_);
        bsize = FFT.bsize;
        
        // data structures for determining the communication partners...
        int  has_length  = (end-start)*pow(points,dim);
        int* has         = new int[has_length];
        int* has_procs   = new int[has_length];
        send_buffer      = new double[has_length*dim];
        send_index       = new int[has_length];
        
        int  want_length = (end_-start_)*pow(n*points,dim-1);
        int* want        = new int[want_length];
        int* want_procs  = new int[want_length];
        recv_buffer      = new double[want_length*dim];
        recv_index       = new int[want_length];

        // S.1: determine all dofs this process posses (+procs)
        for(int ii = start, counter = 0; ii < end; ii++){
            int i = MAP.indices(ii);
            for(int K = 0; K < (dim==3?points:1); K++)
                for(int J = 0; J < points; J++)
                    for(int I = 0; I < points; I++, counter++){
                        // ... determine dof
                        int temp = ((dim==3?((MAP.lbf(i*dim+2)*points+K)*points*n):0) + 
                            MAP.lbf(i*dim+1)*points+J)*points*n+(MAP.lbf(i*dim+0)*points+I);
                        has[counter]       = temp;
                        // ... which process does need this dof?
                        int proc = FFT.indices_proc_rows(temp/pow(n*points,dim-1));
                        has_procs[counter] = proc;
                    }
        }
        
        
        // S.2: which processes needs how many dofs
        std::map<int, int> has_map;
        has_map[rank] = 0; // add this rank
        for(int counter = 0; counter < has_length; counter++){
            int key = has_procs[counter];
            if(has_map.find(key) == has_map.end())
                has_map[key] = 1;
            else
                has_map[key]++;
        }
        
        
        // S.3: determine local indices for collecting data
        for(int i=0;i<has_length;i++)
            send_index[i] = i;
        array_list = has;
        array_proc = has_procs;
        __rank = rank;
        
        qsort(send_index, has_length, sizeof(int), cmp);
        
        
        // S.4: determine offsets
        for(auto& i : has_map)
            if(i.first == rank){
                send_offset.insert(send_offset.begin(), i.second); send_procs.insert(send_procs.begin(), i.first);
            } else {
                send_offset.push_back(i.second); send_procs.push_back(i.first);
            }
        
        send_offset.insert(send_offset.begin(),0);
        
        std::partial_sum(send_offset.begin(), send_offset.end(),send_offset.begin());
        
        
        
        // R.1: determine all dofs this process wants (+procs)
        // ... loop over all local rows
        for(int ii = start_, pn = points*n, counter = 0; ii < end_; ii++)
            // ... loop inside row
            for(int jj = 0; jj < pow(n*points,dim-1); jj++, counter++){
                // ... determine dof
                int temp = ii*pow(n*points,dim-1)+jj;
                want[counter] = temp;
                // ... determine owning process
                int t = dim==3?
                    (temp%pn)/points + ((temp%(pn*pn))/pn)/points*n + (temp/pn/pn)/points*n*n
                    :
                    (temp%pn)/points + (temp/pn)/points*n ;
                int proc = MAP.indices_proc(MAP.indices_inv(t));
                want_procs[counter] = proc;
            }


        // R.2: which processes needs how many dofs
        std::map<int, int> want_map;
        want_map[rank] = 0; // add this rank
        for(int counter = 0; counter < want_length; counter++){
            int key = want_procs[counter];
            if(want_map.find(key) == want_map.end())
                want_map[key] = 1;
            else
                want_map[key]++;
        }

        // R.3: determine local indices for distributing received data
        for(int i=0;i<want_length;i++)
            recv_index[i] = i;
        array_list = want;
        array_proc = want_procs;
        
        qsort(recv_index, want_length, sizeof(int), cmp);
        
        // R.4: add padding
        {
            // consider padding needed by FFTW...
            int N = points * n; int Nx = (N/2+1)*2;
            for(int i = 0; i < want_length;i++)
                recv_index[i] += (recv_index[i]/N)*(Nx-N);
        }
        
        // R.5: determine offsets
        for(auto& i : want_map){
            if(i.first == rank){
                recv_offset.insert(recv_offset.begin(), i.second); recv_procs.insert(recv_procs.begin(),i.first);
            }else{
                recv_offset.push_back(i.second); recv_procs.push_back(i.first);
            }
        }
        
        recv_offset.insert(recv_offset.begin(),0);
        
        std::partial_sum(recv_offset.begin(), recv_offset.end(),recv_offset.begin());
        
        // allocate memory for requests
        recv_requests = new MPI_Request[recv_procs.size()];
        send_requests = new MPI_Request[send_procs.size()];
        
        // print info
        {
#ifdef SIMPLE_VIEW
            char filename[100];
            sprintf(filename, "build/file.%d.out", rank);
            FILE *f = fopen(filename, "w");

            fprintf(f,"Proc %d: \n", rank);

            
#ifdef EXTENDED_VIEW
            for(auto & i : send_offset)
                fprintf(f,"%d ", i);
            fprintf(f,"\n");
        
            fprintf(f,"has:   ");
            for(auto& i : has_map)
                fprintf(f,"%d:%d ", i.first, i.second);
            fprintf(f,"\n");
            
            fprintf(f,"wants: ");
            for(auto& i : want_map)
                fprintf(f,"%d:%d ", i.first, i.second);
            fprintf(f,"\n\n");
#endif
        
            
             
            fprintf(f,"            ");
            if(send_offset[1]!=0)
                fprintf(f,"    ");
            for(int i = 0; i < send_procs.size(); i++){
                fprintf(f,"%4d", send_procs[i]);
                for(int j = 0; j < (send_offset[i+1]-send_offset[i]-1)*5+1; j++)
                    fprintf(f," ");
            }
            fprintf(f,"\n");
            fprintf(f,"has-index:      ");
            for(int counter = 0; counter < has_length; counter++){
                fprintf(f,"%4d ", send_index[counter]);
            }
            
            fprintf(f,"\n");
            fprintf(f,"            ");
            if(recv_offset[1]!=0)
                fprintf(f,"    ");
            for(int i = 0; i < recv_procs.size(); i++){
                fprintf(f,"%4d", recv_procs[i]);
                for(int j = 0; j < (recv_offset[i+1]-recv_offset[i]-1)*5+1; j++)
                    fprintf(f," ");
            }
            fprintf(f,"\n");
            fprintf(f,"wan-index:      ");
            for(int counter = 0; counter < want_length; counter++){
                fprintf(f,"%4d ", recv_index[counter]);
            }
            
#ifdef EXTENDED_VIEW
            fprintf(f,"\n\n");
            
            fprintf(f,"has:   ");

            for(int counter = 0; counter < has_length; counter++){
                fprintf(f,"%4d ", has[counter]);
            }

            fprintf(f,"\n");
            fprintf(f,"       ");

            for(int counter = 0; counter < has_length; counter++){
                fprintf(f,"%4d ", has_procs[counter]);
            }

            fprintf(f,"\n");

            fprintf(f,"wants: ");
            for(int counter = 0; counter < want_length; counter++){
                fprintf(f,"%4d ", want[counter]);
            }
            fprintf(f,"\n");
            fprintf(f,"       ");
            for(int counter = 0; counter < want_length; counter++){
                fprintf(f,"%4d ", want_procs[counter]);
            }
#endif

            fprintf(f,"\n\n");
            fclose(f);
#endif
        }
        
        delete[] has;
        delete[] has_procs;
        delete[] want;
        delete[] want_procs;
    }
    
    /**
     * Destructor
     */
    virtual ~Permutator() {
        // not initialized -> nothing to clean up
        if(!initialized) return;
        
        delete[] recv_index;
        delete[] send_index;
        delete[] send_requests;
        delete[] recv_requests;
    }
    
    /**
     *  Start (nonblocking) permutation:
     *      - write data into buffers for each process and send away data right away
     *      - perform permutation directly on local process
     * 
     * @param source    source array to be permuted...
     * @param target    ...into target array 
     */
    void ipermute(double* source, double* target){
        // save references to vectors (for iwait)
        this->source = source; this->target = target;
        
        // request gathering of data ...
        for(unsigned int i = 1; i < recv_procs.size(); i++)
            MPI_Irecv(recv_buffer+recv_offset[i]*dim, (recv_offset[i+1] - recv_offset[i])*dim,
                MPI_DOUBLE, recv_procs[i], 0, MPI_COMM_WORLD, recv_requests+i-1);
        
        // scatter data ...
        for(unsigned int i = 1; i < send_procs.size(); i++){
            // ... copy data into buffer and ...
            for(int j = send_offset[i]; j < send_offset[i+1]; j++)
                for(int d = 0; d < dim; d++)
                    send_buffer[dim*j+d] = source[send_index[j]*dim+d];

            // ... send right away
            MPI_Isend(send_buffer+send_offset[i]*dim, (send_offset[i+1] - send_offset[i])*dim,
                MPI_DOUBLE, send_procs[i], 0, MPI_COMM_WORLD, send_requests+i-1);
        }
        
        // perform copy operation on local process, if necessary
        for(int s = send_offset[0], t = recv_offset[0]; s < send_offset[1]; s++, t++)
            for(int d = 0; d < dim; d++)
                target[recv_index[t]+d*bsize] = source[dim*send_index[s]+d];
    }
    
    /**
     * Finish (nonblocking) permutation:  wait and write received data from 
     * buffer into target array
     */
    void iwait(){
        // wait for messages...
        for(unsigned int c = 1; c < recv_procs.size(); c++){
            int i = 0;
            // ... a message received: extract index
            MPI_Waitany(recv_procs.size()-1, recv_requests, &i, MPI_STATUSES_IGNORE);
            // ... copy data from buffers
            for(int j = recv_offset[i+1]; j < recv_offset[i+2]; j++)
                for(int d = 0; d < dim; d++)
                    target[recv_index[j]+d*bsize] = recv_buffer[j*dim+d];
        }
        
        // wait that e.th has been send away
        MPI_Waitall(send_procs.size()-1, send_requests, MPI_STATUSES_IGNORE);
    }
    
private:
    // reference to source array
    double* source;
    // ... to target array
    double* target;
    // buffer for sending dofs
    double* send_buffer;
    // buffer for receiving dofs
    double* recv_buffer;
    
    // send: requests
    MPI_Request*     send_requests;
    // ... indices for copying from source into send buffer
    int*             send_index;
    // ... receiving processes
    std::vector<int> send_procs;
    // ... range of buffer to be send to a process
    std::vector<int> send_offset;
    
    // receive: requests
    MPI_Request*     recv_requests;
    // ... indices for copying from recv buffer to target
    int*             recv_index;
    // ... sending processes
    std::vector<int> recv_procs; 
    // ... range of buffer to be received from a process
    std::vector<int> recv_offset;
    
    // dimensions
    int dim;
    // portion of target dedicated for each direction
    int bsize;
};  

}

#endif