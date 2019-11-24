#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"


// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define NROWS 4096
#define NCOLS 4096
#define MASTER 0

void stencil(const int nx, const int ny,int start_col,int end_col, const int width, const int height,
             float **current, float **previous,int rank);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);
int calc_ncols_from_rank(int rank, int size);


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

   // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

 //MPI returns once it has started up processes, gets size and rank
 MPI_Init(&argc,&argv);
 MPI_Comm_size(MPI_COMM_WORLD,&size);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);

 //determining process ranks to left and right of rank
 left = (rank == MASTER) ? (rank+size-1) : (rank-1);
 right = (rank+1)%size;

 local_nrows = NROWS;
 local_ncols = calc_ncols_from_rank(rank,size);
//  printf("ncols is %d and rank is : %d \n",local_ncols,rank);
  // local_ncols = NCOLS/size;
  prev = (float**)malloc(sizeof(float*) * (local_nrows+2));
  for(int ii=0;ii<local_nrows+2;ii++) {
    prev[ii] = (float*)malloc(sizeof(float) * (local_ncols + 2));
  }
  current = (float**)malloc(sizeof(float*) * (local_nrows+2));
  for(int ii=0;ii<local_nrows+2;ii++) {
    current[ii] = (float*)malloc(sizeof(float) * (local_ncols + 2));
  }

    sendbuf = (float*)malloc(sizeof(float) * (local_nrows+2));
  recvbuf = (float*)malloc(sizeof(float) * (local_nrows+2));




  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;

  int start_point= (rank)*(local_nrows+2)*(local_ncols+2);
  int end_point= start_point+ local_ncols;

  // Allocate the image
  float* image = malloc(sizeof(double) * width * height);
  float* image_new = malloc(sizeof(double) * width * height);
  float* tmp_image = malloc(sizeof(double) * width * height);
  float* send_image = malloc(sizeof(float) * height * (local_ncols+2));

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);



    // output_image(OUTPUT_FILE, nx, ny, width, height, image);

    //Split grid between workers
  if (rank == 0){
  for (int j=0;j<local_ncols;j++){
    for (int i=0;i<=local_nrows+1;i++){
      if (i<local_nrows){
      current[i+1][j+1]=(float)image[start_point+((i+1))+(j+1)*height];
      }

      if (i==0 || i==local_nrows+1){
        current[i][j] = 0.0f;
      }

      // if (rank == 1 && (j+1== 255 || j+1==256)){
        // printf("val is %f for j+1 of %d for index: %d\n",image[start_point+((i+1))+(j+1)*height],j+1,start_point+((i+1))+(j+1)*height);
        // printf("starting point is %d",start_point);
      // }
    }
    }
  }
  if (rank == size-1){
        // start_point = rank*local_ncols*height+(local_nrows+2);
          start_point = rank*(NCOLS/size)*height+(local_nrows+2);
                // printf("local n cols is %d for rank %d ",local_ncols,rank);
     for (int j=0;j<local_ncols;j++){
    for (int i=0;i<=local_nrows+1;i++){
      if (i<local_nrows+1){
      current[i][j+1]=(float)image[start_point+((i))+(j)*height];

      // printf("is is : %d, j is %d and val is %f\n",i,j,current[i][j+1]);
      }
       if (i==0 || i==local_nrows+1){
        current[i][j] = 0.0f;
      }
    }
    }

  }
  else{
                    printf("local n cols is %d for rank %d ",local_ncols,rank);

    start_point = rank*local_ncols*height+(local_nrows+2);
     for (int j=0;j<local_ncols;j++){
    for (int i=0;i<=local_nrows+1;i++){
      if (i<local_nrows){
      current[i+1][j+1]=(float)image[start_point+((i+1))+(j)*height];
      }
       if (i==0 || i==local_nrows+1){
        current[i][j] = 0.0f;
      }
    }
    }

  }
    // adding the first rows to send buffer
    for (int ii=0;ii<local_nrows+2;ii++){
      sendbuf[ii] = current[ii][1];
    }
    //sending first row to left and receiving from right
    // MPI_Sendrecv(sendbuf,local_nrows,MPI_FLOAT,left,tag,recvbuf,local_nrows,MPI_FLOAT,right,tag,MPI_COMM_WORLD,&status);
    if (left != size-1){
    MPI_Send(sendbuf,local_nrows+2,MPI_FLOAT,left,tag,MPI_COMM_WORLD);
    }
    if (rank != size-1){
    MPI_Recv(recvbuf,local_nrows+2,MPI_FLOAT,right,tag,MPI_COMM_WORLD,&status);
    for (int ii=0;ii<local_nrows+2;ii++){
      current[ii][local_ncols+1] = recvbuf[ii];
    }
    }
    else {
    for (int ii=0;ii<local_nrows+2;ii++){
    current[ii][local_ncols+1] = 0.0f;
  }
    }

  //addding last row to send buffer
  for (int jj=0;jj<local_nrows+2;jj++){
    sendbuf[jj] = current[jj][local_ncols];
  }
  //Send to the right and receive from left
  if (right != MASTER){
  MPI_Send(sendbuf,local_nrows+2,MPI_FLOAT,right,tag,MPI_COMM_WORLD);
  }
  if (rank != MASTER){
  MPI_Recv(recvbuf,local_nrows+2,MPI_FLOAT,left,tag,MPI_COMM_WORLD,&status);
  for (int ii=0;ii<local_nrows+2;ii++){
    current[ii][0] = recvbuf[ii];
  }
  }
  else {
    for (int ii=0;ii<local_nrows+2;ii++){
    current[ii][0] = 0.0f;
  }
  }
    
  //copy old solution to the u grid
  for (int ii=0;ii<local_ncols+2;ii++){
    for (int jj=0;jj<local_nrows+2;jj++){
      prev[jj][ii] = current[jj][ii];
    }
  }


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

  // printf("here 4 and rank is : %d\n",rank);


  // Call the stencil kernel
  double tic = wtime();
     for (int t = 0; t < niters; ++t) {
    // printf("enters with rank: %d\n",rank);
    stencil(local_nrows,local_ncols,start_col_stencil, end_col_stencil, width, height, current, prev,rank);
    // printf("returned from stencil with rank: %d\n",rank);

       // adding the first rows to send buffer
    for (int ii=0;ii<local_nrows+2;ii++){
      sendbuf[ii] = current[ii][1];
    }
    //sending first row to left and receiving from right
    // MPI_Sendrecv(sendbuf,local_nrows,MPI_FLOAT,left,tag,recvbuf,local_nrows,MPI_FLOAT,right,tag,MPI_COMM_WORLD,&status);
    if (left != size-1){
    MPI_Send(sendbuf,local_nrows+2,MPI_FLOAT,left,tag,MPI_COMM_WORLD);
    }
    if (rank != size-1){
    MPI_Recv(recvbuf,local_nrows+2,MPI_FLOAT,right,tag,MPI_COMM_WORLD,&status);
    for (int ii=0;ii<local_nrows+2;ii++){
      current[ii][local_ncols+1] = recvbuf[ii];
    }
    }
    else {
    for (int ii=0;ii<local_nrows+2;ii++){
    current[ii][local_ncols+1] = 0.0f;
  }
    }

  //addding last row to send buffer
  for (int jj=0;jj<local_nrows+2;jj++){
    sendbuf[jj] = current[jj][local_ncols];
  }
  //Send to the right and receive from left
  if (right != MASTER){
  MPI_Send(sendbuf,local_nrows+2,MPI_FLOAT,right,tag,MPI_COMM_WORLD);
  }
  if (rank != MASTER){
  MPI_Recv(recvbuf,local_nrows+2,MPI_FLOAT,left,tag,MPI_COMM_WORLD,&status);
  for (int ii=0;ii<local_nrows+2;ii++){
    current[ii][0] = recvbuf[ii];
  }
  }
  else {
    for (int ii=0;ii<local_nrows+2;ii++){
    current[ii][0] = 0.0f;
  }
  }
 
  //copy old solution to the u grid
  for (int ii=0;ii<local_ncols+2;ii++){
    for (int jj=0;jj<local_nrows+2;jj++){
      prev[jj][ii] = current[jj][ii];
    }
  }
    //calling stencil a second time
    stencil(local_nrows,local_ncols,start_col_stencil, end_col_stencil, width, height, current, prev,rank);

              // adding the first rows to send buffer
    for (int ii=0;ii<local_nrows+2;ii++){
      sendbuf[ii] = current[ii][1];
    }
    if (left != size-1){
    MPI_Send(sendbuf,local_nrows+2,MPI_FLOAT,left,tag,MPI_COMM_WORLD);
    }
    if (rank != size-1){
    MPI_Recv(recvbuf,local_nrows+2,MPI_FLOAT,right,tag,MPI_COMM_WORLD,&status);
    for (int ii=0;ii<local_nrows+2;ii++){
      current[ii][local_ncols+1] = recvbuf[ii];
    }
    }
    else {
    for (int ii=0;ii<local_nrows+2;ii++){
    current[ii][local_ncols+1] = 0.0f;
  }
    }

  //addding last row to send buffer
  for (int jj=0;jj<local_nrows+2;jj++){
    sendbuf[jj] = current[jj][local_ncols];
  }
  //Send to the right and receive from left
  if (right != MASTER){
  MPI_Send(sendbuf,local_nrows+2,MPI_FLOAT,right,tag,MPI_COMM_WORLD);
  }
  if (rank != MASTER){
  MPI_Recv(recvbuf,local_nrows+2,MPI_FLOAT,left,tag,MPI_COMM_WORLD,&status);
  for (int ii=0;ii<local_nrows+2;ii++){
    current[ii][0] = recvbuf[ii];
  }
  }
  else {
    for (int ii=0;ii<local_nrows+2;ii++){
    current[ii][0] = 0.0f;
  }
  }
    
  for (int ii=0;ii<local_ncols+2;ii++){
    for (int jj=0;jj<local_nrows+2;jj++){
      prev[jj][ii] = current[jj][ii];
    }
  }







 }
    double toc = wtime();


  if (rank == MASTER){
    for (int j=0;j<local_ncols;j++){
      for (int i=0;i<local_nrows;i++){
        image_new[(i+1)+(j+1)*height] = (double)current[i+1][j+1];

      }
    }

    for (int ranks=1;ranks<size;ranks++){
      int start_rank = ranks*local_ncols*height+(local_nrows+2);
            if (ranks == size-1){
          local_ncols=calc_ncols_from_rank(ranks,size);
      }
      for (int j=0;j<local_ncols;++j){
        MPI_Recv(&image_new[(start_rank+1)+(j)*(height)],local_nrows+2,MPI_FLOAT,ranks,tag,MPI_COMM_WORLD,&status);

      }
  }


  output_image("stencil.pgm", nx, ny, width, height, image_new);

  }

    else{
        for (int j=0;j<local_ncols;j++){
          for (int i=0;i<local_nrows;i++){
            send_image[(j+1)*(local_nrows+2)+(i)] = current[i+1][j+1];
          }
        }

        for (int j=1;j<local_ncols+1;j++){
        MPI_Send(&send_image[j*(local_nrows+2)],local_nrows+2,MPI_FLOAT,MASTER,tag,MPI_COMM_WORLD);
        }
  }



  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc - tic);
  printf("------------------------------------\n");


  free(image);
  free(tmp_image);
    MPI_Finalize();


  /* and exit the program */
  return EXIT_SUCCESS;



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
  for (int i=1;i<nx+1;i++){
    for (int j=1;j<ny+1;j++){
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

int calc_ncols_from_rank(int rank, int size)
{
  int ncols;

  ncols = NCOLS / size;       /* integer division */
  if ((NCOLS % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += NCOLS % size;  /* add remainder to last rank */
  }
  
  return ncols;
}
