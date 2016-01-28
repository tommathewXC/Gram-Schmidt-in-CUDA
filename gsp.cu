//----------------------------------------------------------------------------------------------
//Written by: Tom Stephy Mathew
//Class: ECE 641
//Seme: Fall 2014
//----------------------------------------------------------------------------------------------

#include <stdio.h>		//Needed for basic C functionality
#include <math.h>		//Needed for sqrt function
#include "../common/book.h"	//GPU SDK

extern const int N = 3;		//this defines the number of elements in each vector
extern const int M = 3;		//this defines the number of vectors that need to be orthogonalized


//This procedure prints an empty line - used for asthetic reasons
void space(){
	printf("\n");
}

//This procedure prints an array of known length - used for debugging purposes
void printarr(double *m, int size){
	for(int i=0;i<size;i++)
	{
		space();
		printf("%5.2f ", m[i]);
		space();


	}
}


//This procedure prints a matrix of size M x N. If "a" is true, it will be called "Input". Otherwise it will be "output" 
void print_mtx(double *m, bool a){				
	
	if(a){
		
		space();
		printf("Input");
		space();
	
	}
	else{
		space();
		printf("Output");
		space();
	
	}
	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {		
	        printf("%5.2f ", m[i*N+j]);
		}
		space();
    	}
}



//This subroutine computes the dot product of a vector with itself.
//the Input parameter to this procedure is the pointer to the input matrix, the pointer to variable to store the dot product, and the index where the vector starts in the matrix

__global__ void innerprod_self(double *in, double *a, int st){
	__shared__ double prod[N];					//shared memory withing all the threads of this particular block
	int k = threadIdx.x;						//get the indexing for the threads
	prod[k] = in[st + k]*in[k + st];				//compute the element-wise product of the vector across N threads
	__syncthreads();						//wait till all the threads are done with their multiplication process
	
	if(k==0){							//using the first thread to combine the computed pair-wise products
		double temp=0;
		for(int e=0;e<N;e++){
			temp = temp + prod[e];	
		}	
		a[0] = sqrt(temp);					//store the value into the memory pointed to by "a"
	}
}


//This function divides a vector by a known value. 
//The vector starts at "st" of the input, and the scaling coefficient is "val"

__global__ void scale(double *in, double *val, int st){
	int j = threadIdx.x;						// get the thread index
	in[j+st] = in[j+st]/val[0];					// scale over the threads

}


//this subroutine calculates M - i + 1 dot products. These are the dot products of the previous output vector
//with the ith to Mth vectors of the input matrix. The result is stored in "coef", which is of this size.

__global__ void calcCoef(double *v,double *coef, int prev_vindex, int i){
	__shared__ double prod[N];					// this holds the result of the individual dot products
	int r = blockIdx.x;						// this handles the indices for the coefficients
	int t = threadIdx.x;						// this handles the thread indices
	prod[t] = v[prev_vindex + t]*v[N*(r+i) + t];			// calculating the dot products over N threads and M-i+1 blocks
	__syncthreads();
	if(t==0){
		double temp;
		temp  =0;
		for(int e=0;e<N;e++){
			temp = temp + prod[e];	
		}
		coef[blockIdx.x] = -1*temp;				// store the dot products into their approrpiate positions in the coefficient vector
	}
}


//With the coefficient array computed in the subrouting above, we now add M - w + 1 vectors with the M - w + 1 coefficients
//The arrays being combined are output vectors themselves, with the previous input scaled by the (M-w+1)th coefficient
__global__ void combine(double *in, double *coe, int prev, int cur){
	// input parameters: input vector, coefficient array, (i-1)th index, current step index
	int th = threadIdx.x;						//for all N elements
	int bl = blockIdx.x;						//for all the coeficient indices
	int cu = (bl+cur)*N;						//defining where to start adding the vectors from.						
	// ith output = ith input + ith coefficient*previous input
	in[cu + th] = in[cu + th] + coe[bl]*in[prev+th];		//using separate blocks to combine vectors of size N, each which run on N threads
}


int main( void ) {
	double input[9]= {1,0,0,0,2,0,0,0,10};					// defining input. Note you can autofill the input, but there will be no guarantees that it will be linearly independant
	double *dev_input;							// input to be stored on the GPU
	double *dev_m;								// Holds dot product used for normalizing - GPU
	int prev_startindex;							// starting index of the previous vector of the iteration

	//declare pointers
	HANDLE_ERROR(cudaMalloc((void**)&dev_input, (M*N)*sizeof(double)));	// allocate memory for input on GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_m, sizeof(double)));		// allocate memory for dot product result on GPU

	//load data					
	//input = (double*)malloc(M*N*sizeof(double));
	//for(int q =0;q<=M*N;q++)						// uncomment this code to autofill data. Note vectors may not be linearly independent if you do this
	//{
		//input[q] = 10*q;
	//}

	//copy input to GPU
	HANDLE_ERROR(cudaMemcpy(dev_input,input,(M*N)*sizeof(double), cudaMemcpyHostToDevice));

	//print the input matrix
	print_mtx(input,true);

	// iterate through M-1 steps
	for(int w = 1;w<M;w++)				
	{
		double *dev_coef;							// pointer to hold coeffieicnts on GPU
		int num_of_coe;
		num_of_coe = M - w;							// the number of coeffieicnts needed for the wth iteration
		HANDLE_ERROR(cudaMalloc((void**)&dev_coef, (M-w)*sizeof(double)));	// declare space on the GPU for the coefficeints
		prev_startindex = (w-1)*N;						// compute the index of the previous vector in reference to this iteration


		// Normalizing (i-1)th output vector	
		innerprod_self<<<1,N>>>(dev_input,dev_m, prev_startindex);		// compute the dot product of the previous vector with itself
		scale<<<1,N>>>(dev_input,dev_m,prev_startindex);			// use the computed dot product to scale the vector
	

		// coefficient code		
		calcCoef<<<num_of_coe,N>>>(dev_input, dev_coef, prev_startindex,w);	// compute the M-w+1 coeffieints for this iteration



		// combine vectors

		combine<<<num_of_coe,N>>>(dev_input, dev_coef, prev_startindex, w);	// using the coefficients, partially alter M-w+1 vectors in the input matrix.


		HANDLE_ERROR(cudaFree(dev_coef));					// free the memory for the coefficients
	
	}

	prev_startindex = (M-1)*N;							// this and the next three lines are used to normalize the last output
	innerprod_self<<<1,N>>>(dev_input,dev_m, prev_startindex);
	scale<<<1,N>>>(dev_input,dev_m,prev_startindex);
	
	//copy result back to CPU
	HANDLE_ERROR(cudaMemcpy(input, dev_input, M*N*sizeof(double), cudaMemcpyDeviceToHost));

	print_mtx(input, false);



	//Free pointers
	HANDLE_ERROR(cudaFree(dev_input));
	HANDLE_ERROR(cudaFree(dev_m));
	//free(input);									// uncomment this line if you autofill the input

	return 0;
}

