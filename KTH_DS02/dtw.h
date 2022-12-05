#ifndef DTW_H_INCLUDED
#define DTW_H_INCLUDED

double dtw(double *x, double *y, int xsize, int ysize, int window);

double LB_Keogh(double *x, double *y, int xsize, int ysize, int bound);

#endif // DTW_H_INCLUDED
