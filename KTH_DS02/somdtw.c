#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "somdtw.h"
#include "dtw.h"

#define VINF 1e15



void init_conf(int channel, int height, int width, int n_iter, int w, int b, double lr, double sigma, t_conf *NetConf)
{
    NetConf->channel = channel;
    NetConf->height = height; // row, x
    NetConf->width = width;  // column, y
    NetConf->neuron = height * width;
    NetConf->n_iter = n_iter;
    NetConf->w = w;
    NetConf->b = b;
    NetConf->lr = lr;
    NetConf->sigma = sigma;
}

/*
void alloc_array_struct(int n)
{
    d_array=malloc(n*sizeof(Data));
    int i;
    for(i=0;i<n;i++)
    {
        d_array[i].data=malloc(N_conf.n_feature*sizeof(double));
        d_array[i].cluster=malloc(sizeof(int));
    }
}
*/

double euc_dist1(double *x, double *y, int xlen)
{
    double sum = 0.0;
    int i;
    for(i = 0; i < xlen; ++i)
    {
        sum += pow((x[i]-y[i]),2.0);
    }
    return sqrt(sum);
}


double* init_rand_weight(double rg, int ch)
{
    int i;
    double k = (double) rand()/RAND_MAX;
    double *tmp_w = malloc(ch*sizeof(double));

    for(i=0; i<ch; i++)
    {
        tmp_w[i] = k * rg + rg/2;
    }

    double norm=0.;

    for(i=0; i<ch; i++)
    {
        norm += pow(tmp_w[i], 2.0);
    }

    for(i=0; i<ch; i++)
    {
            tmp_w[i] /= norm;
    }
    return tmp_w;

}

//array shuffle

void init_shuffle(int *index_array, int n)
{
    int i;
    for(i=0;i<n;i++)
    {
        index_array[i] = i;
    }
    return;
}


void array_shuffle(int *index_array, int n)
{
    int i,ri,k;
    srand(time(NULL));
    for(i=n-1; i>=1; i--)
    {
        ri = (int) rand() % i;
        k = index_array[i];
        index_array[i] = index_array[ri];
        index_array[ri] = k;
    }

    return;
}


// create and initialize map
void create_neuron_map(t_net *Net, t_conf *NetConf)
{

    int i, j;
    Net->map = malloc(NetConf->height * sizeof(t_node *));
	for(i=0; i<NetConf->height; i++)
	{
		Net->map[i] = malloc(NetConf->width * sizeof(t_node));
	}
	for(i=0; i<NetConf->height; i++)
	{
		for (j=0; j<NetConf->width; j++)
		{
            Net->map[i][j].w = (double*)malloc(sizeof(double)*NetConf->channel);
			Net->map[i][j].w = init_rand_weight(0.005, NetConf->channel);
		}
	}
	Net->nb_size = NetConf->sigma;

}

void update(t_bmu *bn, t_net *Net, t_conf *NetConf)
{
    int ns = Net->nb_size;
    int i,j,k,x1,y1,x2,y2; // bubble size
    for(; ns>=0; ns--)
    {
        if(bn->r - ns < 0) x1 = 0;
        else x1 = bn->r - ns;
        if(bn->c - ns < 0) y1 = 0;
        else y1 = bn->c - ns;

        if(bn->r + ns > NetConf->height - 1) x2 = NetConf->height - 1;
        else x2 = bn->r + ns;
        if(bn->c + ns > NetConf->width - 1) y2 = NetConf->width - 1;
        else y2 = bn->c + ns;
        // bubble update
        //printf("i: %d; j: %d; k: %d. \n", x2 ,y2, x1);
        for(i=x1; i<=x2; i++)
        {
            for(j=y1; j<=y2; j++)
            {
                for(k=0; k<NetConf->channel; k++)
                {
                    Net->map[i][j].w[k] += Net->lr * (Net->data[k] - Net->map[i][j].w[k]);
                }
            }
        }
    }
    return;
}


double lr_decay(double olr, int it, int n_it)
{
    double lr;
    lr = olr / (1.0 + (double)it/((double)n_it/2));
    return lr;
}


int nb_decay(double sigma, int it, int n_it)
{
    double p;
    int k;
    p = sigma / (1.0 + (double)it/((double)n_it/2));
    if (p > 1.0)
    {
        k = (int) floor(p);
    }
    else k = 1;

    return k;
}


void train(t_bmu *classification, double *data, int n_samples, int m_features, t_net *Net, t_conf *NetConf, int mode)
{
    double dist, min_dist, total_dist;
    int i, j, it, s, idx;
    int cx, cy;
    t_bmu *BMU = malloc(sizeof(t_bmu));

    int *index_array = malloc(sizeof(int)*n_samples);

    init_shuffle(index_array, n_samples);
    double t1 = 0;
    for(it=0; it<NetConf->n_iter; it++)
    {
        total_dist = 0;
        Net->lr = lr_decay(NetConf->lr, it, NetConf->n_iter);
        Net->nb_size = nb_decay(NetConf->sigma, it, NetConf->n_iter);
        array_shuffle(index_array, n_samples);

        clock_t t_start,t_end;
        t_start = clock();

        for(s=0; s<n_samples; s++)
        {
            idx = index_array[s];
            Net->data = data + idx * m_features;
            min_dist = VINF;
            cx = 0;
            cy = 0;
            // find bmu
            //printf("%d %d abc.\n", NetConf->height, NetConf->width);
            //printf("%d %d def.\n", NetConf->n_iter, s);
            if(mode == 1){
                for(i=0; i < NetConf->height; i++)
                {
                    for(j=0; j < NetConf->width; j++)
                    {
                        //printf("%d %d abc.\n", i, j);
                        dist = LB_Keogh(Net->data, Net->map[i][j].w, m_features, m_features, NetConf->b);
                        if(dist < min_dist)
                        {
                            dist = dtw(Net->data, Net->map[i][j].w, m_features, m_features, NetConf->w);
                            if(dist < min_dist)
                            {
                                min_dist = dist;
                                cx = i;
                                cy = j;
                            }
                        }

                    }
                }
            }
            else if(mode == 2){
                for(i=0; i < NetConf->height; i++)
                {
                    for(j=0; j < NetConf->width; j++)
                    {
                        //printf("%d %d abc.\n", i, j);
                        dist = euc_dist1(Net->data, Net->map[i][j].w, m_features);
                        if(dist < min_dist)
                        {
                            min_dist = dist;
                            cx = i;
                            cy = j;
                        }

                    }
                }
            }
            else
            {
                printf("Wrong mode.\n");
                free(BMU);
                free(index_array);
                return;
            }

            BMU->r = cx;
            BMU->c = cy;
            BMU->dist = min_dist;
            total_dist += min_dist;
            classification[idx].r = cx;
            classification[idx].c = cy;
            classification[idx].dist = min_dist;

        }
        t_end = clock();
        printf("Time=%f, cycle = %d.\n",(double)(t_end-t_start)/CLOCKS_PER_SEC, it+1);
        t1 += (double)(t_end-t_start)/CLOCKS_PER_SEC;
        update(BMU, Net, NetConf);
        total_dist /= n_samples;
        printf("Total distance in %d iteration is %f.\n", it+1, total_dist);


    }
    printf("Time=%f.\n",(double)(t1/NetConf->n_iter));
    free(BMU);
    free(index_array);
    return;
}


void somtrain(t_bmu *classification, double *data, int n_samples, int m_features, int height,
              int width, int n_iter, int w, int b, double lr, double sigma, int mode)
{
     t_conf *NetConf = malloc(sizeof(t_conf));
     t_net *Net = malloc(sizeof(t_net));
     init_conf(m_features, height, width, n_iter, w, b, lr, sigma, NetConf);
     printf("Initialization.\n");
     create_neuron_map(Net, NetConf);
     printf("Create map.\n");
     train(classification, data, n_samples, m_features, Net, NetConf, mode);
     printf("Training process complete.\n");
     free(NetConf);
     free(Net);
     return;

}



/*
int main()
{
    double data[2][128] = {
        {0.037764633,2.5311236E-4,0.003977076,0.020764058,0.02173118,0.114,0.414,0.718,0.889,0.969,1.002,1.014,1.017,0.905,0.605,0.304,0.138,0.061,0.053973303,0.0146603,0.035561278,0.045494101,0.030238212,0.043190555,0.021148818,0.039419722,0.017780126,0.04535716,0.04564139,0.018776462,0.012347944,0.02442172,0.018924466,0.0069161007,0.042104272,5.1942129E-4,-0.114,-0.399,-0.682,-0.836,-0.906,-0.933,-0.943,-0.944,-0.944,-0.948,-0.949,-0.946,-0.948,-0.951,-0.952,-0.952,-0.953,-0.952,-0.952,-0.952,-0.95,-0.949,-0.952,-0.956,-0.96,-0.961,-0.959,-0.958,-0.96,-0.961,-0.96,-0.96,-0.959,-0.958,-0.956,-0.953,-0.949,-0.947,-0.948,-0.951,-0.95,-0.945,-0.941,-0.941,-0.943,-0.944,-0.943,-0.944,-0.952,-0.957,-0.953,-0.946,-0.838,-0.56,-0.284,-0.131,-0.057,-0.015058931,0.0059390887,0.0022057233,0.0083295739,0.028908968,0.0049785927,0.041239881,0.010012772,0.019281287,0.033087255,0.0099204681,0.026583031,0.0094319071,0.018443621,0.023889741,0.043350606,0.030228536,0.031086542,0.012996399,0.034532338,0.03117658,0.019603584,0.027655279,0.041001638,0.019075367,0.034208601,0.022830972,0.013387386,0.038906948,0.039187692,0.053616705,0.054673016,0.031523157,0.03978737,0.020660802},
        {0.041747222,0.026262035,0.0024890674,0.023925895,0.039603507,0.039008461,0.021147631,0.018765444,0.026405114,0.020522961,0.023448213,0.0018729446,0.003766064,0.030746577,0.04354786,0.034775415,0.046251109,0.040955577,0.018543766,0.016918974,0.025676501,0.020744765,0.01273871,0.048883031,0.010178469,0.0087539087,0.020608857,0.026569593,0.038964386,0.0020765946,0.026232962,0.010357591,0.01102224,0.026302193,0.032369582,0.0439846,0.032001582,0.013770521,0.032339293,0.0099025671,0.117,0.423,0.731,0.902,0.979,1.013,1.027,1.03,1.032,1.036,1.035,1.035,1.036,1.037,1.04,1.043,1.043,1.04,1.039,1.041,1.04,1.036,1.035,1.034,1.031,1.031,1.034,1.039,1.041,1.039,1.034,1.029,1.029,1.033,1.035,1.036,1.034,1.034,1.04,0.927,0.618,0.308,0.137,0.057,0.052822401,0.03712816,0.028541798,0.025232772,0.05503598,0.015742396,0.011606101,0.039391864,0.03830194,0.049148719,0.019788488,0.023780675,0.049672354,0.037647625,0.050046987,0.048581181,0.026163674,0.029072628,0.026550945,0.038327926,0.032552285,0.052308944,0.035750869,0.03598284,0.027195561,0.010571572,7.1233322E-4,0.036944592,0.0045633801,0.039512027,0.042119848,0.04212316,0.035241181,0.0057523678,0.028412641,0.040423775,0.028007071,0.042979278,0.0078219738,0.013169194,0.022393536,0.037013136,0.030751863,0.053828632}
    };
    double *p = data;
    init_conf(channel=128, height=3, width=3, n_iter=10, w=15, b=5, lr=0.5, sigma=3.0);
    create_neuron_map();
    t_bmu *cl = NULL;
    cl = train(p, 2, 128);

    printf("Over.\n");

	return 0;
}
*/
