#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "dtw.h"

// DTW in C

double dtw(double *x, double *y, int xsize, int ysize, int window)
{
    // calculate the distance between list x and list y
    const double VINF = 1e12;
    const int min_window = abs(xsize - ysize);
    int i, j, minj, maxj, w;
    double dist, min;
    // create distance matrix (x+1)*(y+1)
    double **distances = malloc((xsize + 1) * sizeof(double *));
    for(i = 0; i < xsize + 1; ++i)
        distances[i] = malloc((ysize + 1) * sizeof(double));

    w = window;

    if(w < min_window)
        w = min_window;

    for(i = 0; i <= xsize; ++i)
        for(j = 0; j <= ysize; ++j)
            distances[i][j] = VINF;
    distances[0][0] = 0.0;

    for(i = 0; i < xsize; ++i)
    {
        minj = i - w;
        if(minj < 0)
            minj = 0;
        maxj = i + w;
        if(maxj > ysize)
            maxj = ysize;
        for(j = minj; j < maxj; ++j)
        {
            dist = pow((x[i] - y[j]), 2.0);
            min = distances[i][j];
            if(min > distances[i][j+1])
                min = distances[i][j+1];
            if(min > distances[i+1][j])
                min = distances[i+1][j];
            distances[i+1][j+1] = dist + min;
        }
    }

    dist = distances[xsize][ysize];

    for(i = 0; i < xsize + 1; ++i)
        free(distances[i]);
    free(distances);

    return sqrt(dist);

}


double min(double *x, int start, int end)
{
    double tmp = x[start];
    for(int i = start; i < end; i++)
    {
        if(tmp > x[i]) tmp = x[i];
    }
    return tmp;
}

double max(double *x, int start, int end)
{
    double tmp = x[start];
    for(int i = start; i < end; i++)
    {
        if(tmp < x[i]) tmp = x[i];
    }
    return tmp;
}



double LB_Keogh(double *x, double *y, int xsize, int ysize, int bound)
{
    double lbsum = 0.0;
    int i, minj, maxj, b;
    double lowerbound;
    double upperbound;
    b = bound;
    for(i = 0; i < xsize; ++i)
    {
        minj = i - b;
        if(minj < 0) minj = 0;
        maxj = i + b;
        if(maxj > ysize) maxj = ysize;
        lowerbound = min(y, minj, maxj);
        upperbound = max(y, minj, maxj);
        if(x[i] > upperbound) lbsum += pow((x[i]-upperbound), 2.0);
        else if (x[i] < lowerbound) lbsum += pow((x[i]-lowerbound), 2.0);
    }

    return sqrt(lbsum);

}
