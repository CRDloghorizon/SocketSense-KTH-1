#ifndef SOMDTW_H_INCLUDED
#define SOMDTW_H_INCLUDED


typedef struct N_config
{
    int channel; //data vector size (feature numbers)
    int height; // neuron map row number
    int width; // neuron map column number
    int neuron; //total neuron r * c
    int n_iter; // iteration number
    int w; // window
    int b;  // bound
    double lr; // learning rate
    double sigma; // neighbor size bubble
} t_conf;


typedef struct node
{
    double dist; // dtw distance
    double *w; // weight vector
} t_node;


typedef struct somnet //som net
{
    int nb_size;  // current neighborhood size
    t_node **map;  // map
    double *data; // current data vector
    double lr; // current lr
} t_net;


typedef struct bmu
{
    double dist; // dtw distance
    int r;  // row
    int c; // column
} t_bmu;



void somtrain(t_bmu *classification, double *data, int n_samples, int m_features, int height, int width, int n_iter, int w, int b, double lr, double sigma, int mode);



#endif // SOMDTW_H_INCLUDED
