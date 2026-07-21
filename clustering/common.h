#ifndef COMMON_H
#define COMMON_H

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include "gdal.h"
#include "cpl_conv.h"

#define MAX_K               12
#define MAX_BANDS           16
#define BLOCK_SIZE          256
#define NUM_CLASS_COLORS    12
#define MIN_COVAR           1e-3f

extern const int CLASS_COLORS[NUM_CLASS_COLORS][3];

typedef struct {
    float *data;            // n_pixels * n_bands
    int n_pixels;
    int n_bands;
    int has_nodata;         // 1 si hay valor NoData
    float nodata_value;     // valor NoData (mismo en todas las bandas)
} Image;

typedef struct {
    int init;
    int max_iter;
    float convergence;
} KMeansConfig;

typedef struct {
    float *centroids;   // k * n_bands
    int *labels;        // n_pixels
    int k;
    int n_bands;
} KMeansModel;

typedef struct {
    int kmeans_init_iter;
    int max_iter;
    float convergence;
} GMMConfig;

typedef struct {
    float *mean;        // media
    float *covar;       // matriz de covarianzas
    float *covar_inv;   // matriz de covarianzas inversa
    float covar_det;    // determinante de la matriz de covarianzas
    float priori;       // probabilidad a priori
} GaussianComponent;

typedef struct {
    GaussianComponent *components;  // gaussianas
    float *post;                    // probabilidades a posteriori
    int *labels;
    int k;
    int n_bands;
} GMMModel;

#ifdef __cplusplus
extern "C" {
#endif

void lcg_set_seed(unsigned int seed);
unsigned int lcg_rand(void);

Image *load_image(const char *filepath);
void free_image(Image *img);
void save_image(const int *labels, int n_pixels, int k, const char *output_path, const char *reference_path);
void save_csv(const int *labels, int n_pixels, const char *filepath);
float *transpose_to_band_major(const Image *img, size_t *padded_pixels);
int get_random_pixel(const Image *img);

void print_cluster_sizes(const int *labels, int n_pixels, int k);
void print_centroids(const KMeansModel *model);
void print_components(const GMMModel *model);

void init_centroids_random(KMeansModel *model, const Image *img);
void init_centroids_pp(KMeansModel *model, const Image *img);
void init_params_kmeans(GMMModel *model, const Image *img, int init_iter);

float euclidean_distance_sq(const float *a, const float *b, int n);
int cholesky_decompose(const float *A, float *L, int n);
void cholesky_invert(const float *L, float *A_inv, int n);
float cholesky_determinant(const float *L, int n);
void compute_covar_derived(GaussianComponent *g, int n_bands);

KMeansModel *kmeans_init(int k, int n_bands);
void kmeans_free(KMeansModel *model);
GMMModel *gmm_init(int k, int n_bands);
void gmm_free(GMMModel *model);

#ifdef __cplusplus
}
#endif

static inline int is_nodata(const float *pixel, const Image *img)
{
    if (!img->has_nodata) return 0;
    for (int b = 0; b < img->n_bands; b++)
        if (pixel[b] != img->nodata_value) return 0;
    return 1;
}

#endif // COMMON_H