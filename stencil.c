#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define NROWS 1024
#define NCOLS 1024
#define MASTER 0

void stencil(const int nx, const int ny,int start_col,int end_col, const int width, const int height,
             float **current, float **previous,int rank);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
int calc_ncols_from_rank(int rank, int size);
double wtime(void);

int main(int argc, char* argv[])
{
 int rank; //rank of this process
 int left; //rank of process to left
 int right; //rank of process to right
 int size; //number of processes in communicator
 int tag = 0; //scope for adding extra information to a message 
 MPI_Status status; //Struct used by MPI_RECV
 int local_nrows; //number of rows for this rank
 int local_ncols; //number of cols for this rank
 int remote_ncols; //number of cols given to this remote rank
 float **prev; //previous grid;
 float **current; //current grid
 float *sendbuf; //buffer to hold values to send
 float *recvbuf; //buffer to hold received values
 float *printbuf; //buffer to hold values for printing
 
 //MPI returns once it has started up processes, gets size and rank
 MPI_Init(&argc,&argv);
 MPI_Comm_size(MPI_COMM_WORLD,&size);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);

 //determining process ranks to left and right of rank
 left = (rank == MASTER) ? (rank+size-1) : (rank-1);
 right = (rank+1)%size;

 local_nrows = 1024;
 local_ncols = 256;

  prev = (float**)malloc(sizeof(float*) * (local_nrows));
  for(int ii=0;ii<local_nrows;ii++) {
    prev[ii] = (float*)malloc(sizeof(float) * (local_ncols + 2));
  }
  current = (float**)malloc(sizeof(float*) * (local_nrows));
  for(int ii=0;ii<local_nrows;ii++) {
    current[ii] = (float*)malloc(sizeof(float) * (local_ncols + 2));
  }
  sendbuf = (float*)malloc(sizeof(float) * local_nrows);
  recvbuf = (float*)malloc(sizeof(float) * local_nrows);
  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */ 
  remote_ncols = calc_ncols_from_rank(size-1, size); 
  printbuf = (float*)malloc(sizeof(float) * (local_ncols + 2));
  
  //defined start point and end point for cols to break up image
  int start_point= (rank)*local_ncols;
  int end_point= start_point+ local_ncols;
  // printf("%d\n",start_point_cols);
  // printf("%d\n",end_point_cols);



  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;

  // Allocate the image
  float* image = malloc(sizeof(double) * width * height);

  float* tmp_image = malloc(sizeof(double) * width * height);

  // Set the input image
  init_image(local_nrows, local_ncols, width, height, image, tmp_image);

  // printf("local cols : %d",local_ncols);

  printf("height: %d\n",height);
  int height_new = 256;
    printf("here 0\n");



  //Split grid between workers
  for (int i=0;i<=local_ncols;i++){
    for (int j=0;j<local_nrows;j++){
      // printf("i is : %d and j is : %d\n",i,j);
      current[j][i+1]=(float)image[(start_point+i+1)+(j+1)*height];
      // if (j==1023 & i==2){printf("copied");}
      // printf("%f ", current[i][j]);

    }
    // printf("\n");
    // printf("one batch fin for %d\n\n",rank);

  }
  printf("here 1\n");
  // printf("%f\n",current[1024][2]);

  //adding the first rows to send buffer
    for (int ii=0;ii<local_nrows;ii++){
      sendbuf[ii] = current[ii][1];
    }
    //sending first row to left and receiving from right
    MPI_Sendrecv(sendbuf,local_nrows,MPI_FLOAT,left,tag,recvbuf,local_nrows,MPI_FLOAT,right,tag,MPI_COMM_WORLD,&status);
    for (int ii=0;ii<local_nrows;ii++){
      current[ii][local_ncols+1] = recvbuf[ii];
    }

  //addding last row to send buffer
  for (int jj=0;jj<local_nrows;jj++){
    sendbuf[jj] = current[jj][local_nrows];
  }
  //Send to the right and receive from left
  MPI_Sendrecv(sendbuf,local_nrows,MPI_FLOAT,right,tag,recvbuf,local_nrows,MPI_FLOAT,left,tag,MPI_COMM_WORLD,&status);
  for (int ii=0;ii<local_nrows;ii++){
    current[ii][0] = recvbuf[ii];
  }
    printf("here 2\n");
  // printf("proof: %f\n",current[1024][2]);

  //copy old solution to the u grid
  for (int ii=0;ii<local_ncols+2;ii++){
    for (int jj=0;jj<local_nrows;jj++){
      prev[jj][ii] = current[jj][ii];
    }
      
  }
  printf("this value is present %f\n",prev[1023][2]);

      // printf("%f\n",current[1024][256]);
    printf("here 3\n");


  //coming up with start and end columns for stencil function

  int start_col_stencil;
  int end_col_stencil;

  if (rank==0){
    start_col_stencil = 2;
    end_col_stencil = local_ncols;
  }
  else if (rank == size-1){
    start_col_stencil = 1;
    end_col_stencil = local_ncols-1;
  }
  else{
    start_col_stencil = 1;
    end_col_stencil = local_ncols;
  }

  printf("here 4 and rank is : %d\n",rank);

  
  
  
  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < 1; ++t) {
    // printf("enters with rank: %d\n",rank);
    // stencil(nx,ny,start_col_stencil, end_col_stencil, width, height, current, prev,rank);
    // printf("returned from stencil with rank: %d\n",rank);

    //redoing halos
    //adding the first rows to send buffer
  //   for (int ii=0;ii<local_nrows;ii++){
  //     sendbuf[ii] = current[ii][1];
  //   }
  //   //sending first row to left and receiving from right
  //   MPI_Sendrecv(sendbuf,local_nrows,MPI_FLOAT,left,tag,recvbuf,local_nrows,MPI_FLOAT,right,tag,MPI_COMM_WORLD,&status);
  //   for (int ii=0;ii<local_nrows;ii++){
  //     current[ii][local_ncols+1] = recvbuf[ii];
  //   }

  // //addding last row to send buffer
  // for (int jj=0;jj<local_nrows;jj++){
  //   sendbuf[jj] = current[jj][local_nrows];
  // }
  // //Send to the right and receive from left
  // MPI_Sendrecv(sendbuf,local_nrows,MPI_FLOAT,right,tag,recvbuf,local_nrows,MPI_FLOAT,left,tag,MPI_COMM_WORLD,&status);
  // for (int ii=0;ii<local_nrows;ii++){
  //   current[ii][0] = recvbuf[ii];
  // }
    //after calling stencil function once ,copying current into previous 
      for (int ii=0;ii<local_ncols+2;ii++){
    for (int jj=0;jj<local_nrows;jj++){
      prev[jj][ii] = current[jj][ii];
    }
          

    //calling stencil a second time
    // stencil(nx,ny,start_col_stencil, end_col_stencil, width, height, current, prev,rank);



      }
      printf("here 5\n");


  }

  //collecting results for master 
  //dealing with first worker

    printf("here 6 with rank %d\n",rank);

  if (rank == MASTER){
    for (int j=0;j<=local_ncols;j++){
      for (int i=0;i<local_nrows;i++){
        image[(j+1)+(i+1)*height] = current[i][j+1];
        // printf("rows is : %d and cols is : %d and value is %f \n",i,j,image[(j)*(i)*height]);

      }
    }
          // printf("%f\n",current[1024][256]); //DEADLOCK

  
    printf("RANK AFTER 6 %d",rank);
    

    for (int ranks=1;ranks<size;ranks++){
      for (int jj=0;jj<local_nrows;jj++){
      MPI_Recv(printbuf,local_ncols+2,MPI_FLOAT,ranks,tag,MPI_COMM_WORLD,&status);
      current[jj] = printbuf;
      }
      printf("Receiving from rank: %d from MASTER\n",ranks);
      if (ranks == 3){
        for (int j=0;j<=local_ncols;j++){
        for (int i=0;i<local_nrows;i++){
          image[(start_point+j+1)+(i+1)*height] = current[i][j+1];
        }
      }
      }
      else {
      for (int j=0;j<=local_ncols;j++){
        for (int i=0;i<local_nrows;i++){
          image[(start_point+j+1)+(i+1)*height] = current[i][j+1];
        }
      }
      }
    }
    }

  else {
                // printf("%f\n",current[1024][256]);
        // image[((start_point+j+1)+(i+1)*height)] = current[i][j];
        for (int ii=0;ii<local_nrows;ii++){
        MPI_Send(current[ii],local_ncols+2,MPI_FLOAT,MASTER,tag,MPI_COMM_WORLD);
        }
        printf("Sending from rank: %d to MASTER\n",rank);

  }
    double toc = wtime();
      printf("here 9");


//  for (int ii=0;ii<ny+1;ii++){
//     for (int jj=0;jj<nx;jj++){
//       printf("%f",image[ii+jj*height]);
//     }
//     printf("\n");
//   }


  

  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc - tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, width, height, image);
  free(image);
  free(tmp_image);
}

void stencil(const int nx, const int ny,int start_col,int end_col,const int width, const int height,
             float **current, float **prev,int rank)
{
  // for (int i = 1; i < nx + 1; ++i) {
  //   for (int j = 1; j < ny + 1; ++j) {
  //     tmp_image[j + i * height] =  image[j     + i       * height] * 0.6f
  //      + (image[j     + (i - 1) * height] 
  //     + image[j     + (i + 1) * height] 
  //     + image[j - 1 + i       * height] 
  //     + image[j + 1 + i       * height]) * 0.1f;
  //   }
  // }
  for (int i=1;i<nx-1;i++){
    for (int j=start_col;j<=end_col;j++){
      // printf("I = %d and j=%d for rank %d with prev value: %f \n",i,j,rank,prev[i][j]);
      // printf("%f\n")
      // if (i==1023 && j==3){printf("HOLD");}
      current[i][j] = prev[i][j]*0.6f+(prev[i-1][j] + prev[i+1][j] + prev[i][j-1]+ prev[i][j+1])*0.1f;
    }
  }

}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int calc_ncols_from_rank(int rank,int size){
 int ncols;
 ncols = NCOLS/size;
 if ((NCOLS%size) !=0){
 if (rank == size-1)
	ncols +=NCOLS%size;
}
 return ncols;

} 
