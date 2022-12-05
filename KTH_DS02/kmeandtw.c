#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "kmeandtw.h"
#include "dtw.h"

#define VINF 1e15

double euc_dist(double *x, double *y, int xlen)
{
    double sum = 0.0;
    int i;
    for(i = 0; i < xlen; ++i)
    {
        sum += pow((x[i]-y[i]),2.0);
    }
    return sqrt(sum);
}



void kmeans(int *assignment, int K, int max_iter, int w, int b, int n_samples, int m_features, double *data, int mode)
{
    // head pointer of each sample = data + i*m_features
    // return value assignment = int[n_samples]
    int count,i,j,n;
    int cen_ind[K];
    double min_dist, dist;
    int closest_cen;

    if(K > n_samples)
    {
        printf("Error: too many clusters.\n");
        return;
    }



    double centroids[K][m_features];
    // random initialize the centroids
    srand(time(NULL));
    int im, in;
    int cen[n_samples];
    for(i=0; i<n_samples; i++) cen[i]=i;
    for(i=n_samples-1; i>=1; --i) {
        im = (int) rand()%i;
        in = cen[i];
        cen[i] = cen[im];
        cen[im] = in;}

    for(i=0;i<K;i++) cen_ind[i] = cen[i];

    for(i=0;i<K;i++)
    {
        for(j=0;j<m_features;j++)
        {
            //printf("data %d:%d\n", cen_ind[i]*n_samples,j); !!!!
            centroids[i][j] = data[cen_ind[i]*m_features+j];
        }
    }

    double *p = centroids; // use flatten pointer
    count = 0;
    double t1 = 0;
    for(n=0; n < max_iter; n++)
    {
        count++;
        printf("Kmeans iter: %d.\n", count);
        // cluster

        clock_t t_start,t_end;
        t_start = clock();

        for(i=0; i < n_samples; i++)
        {
            min_dist = VINF;
            closest_cen = 0;
            if (mode == 1){
                for(j=0; j < K; j++)
                {
                    dist = LB_Keogh(data+i*m_features, p+j*m_features, m_features, m_features, b);
                    if(dist < min_dist)
                    {
                        dist = dtw(data+i*m_features, p+j*m_features, m_features, m_features, w);
                        if(dist < min_dist)
                        {
                            min_dist = dist;
                            closest_cen = j;
                        }
                    }

                }
            }
            else if(mode == 2){
                for(j=0; j < K; j++)
                {

                    dist = euc_dist(data+i*m_features, p+j*m_features, m_features);
                    //printf("number %d\n",i+1);
                    if(dist < min_dist)
                    {
                        min_dist = dist;
                        closest_cen = j;
                    }

                }
            }
            else
            {
                printf("Wrong mode.\n");
                return;
            }


            assignment[i] = closest_cen;
        }
        t_end = clock();


        printf("Time=%f, cycle = %d.\n",(double)(t_end-t_start)/CLOCKS_PER_SEC, n+1);
        t1 += (double)(t_end-t_start) / CLOCKS_PER_SEC;
        // update
        memset(centroids, 0, sizeof(centroids));
        memset(cen_ind, 0, sizeof(cen_ind));
        for(i=0;i < n_samples; i++)
        {
            for(j=0; j < m_features; j++)
            {
                centroids[assignment[i]][j] += data[i*m_features + j];
            }
            cen_ind[assignment[i]] += 1;
        }
        for(i=0; i < K; i++)
        {
            for(j=0; j < m_features; j++)
            {
                centroids[i][j] = centroids[i][j] / cen_ind[i];
            }
        }

    }
    printf("Time=%f.\n",(double)(t1/max_iter));

    // assignment is a n_sample length int array contains the cluster
    return;
}
