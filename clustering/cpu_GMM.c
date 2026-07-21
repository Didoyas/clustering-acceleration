/*
 * cpu_GMM.c
 * Gaussian Mixture Model - CPU Version
 *
 * Compile (Linux):
 * Compile with: gcc -O2 -o cpuGMM cpu_GMM.c common.c -lm $(gdal-config --cflags) $(gdal-config --libs)
 *      or with: gcc -O2 -o cpuGMM cpu_GMM.c common.c -lm -lgdal
 *
 * Compile (MSVC on Windows / OSGeo4W):
 * Compile with: cl cpu_GMM.c common.c /O2 /I"C:\OSGeo4W\apps\gdal-dev\include" /link /LIBPATH:"C:\OSGeo4W\apps\gdal-dev\lib" gdal_i.lib /out:cpuGMM.exe
 *
 * Run:
 *   ./cpuGMM <image_filepath> <k> [max_iter] [convergence] [seed] [k-means_init_iter]
 *   Example: ./cpuGMM sample.tif 4 100 0.001 12345 10
 */

#include "common.h"

double      e_step(GMMModel *model, const Image *img);
float       m_step(GMMModel *model, const Image *img, double **new_means, double **new_covars, float **old_means);

float       log_gaussian(const float *x, const GaussianComponent *comp, int n_bands);
int         *get_hard_labels(const GMMModel *model, const Image *img, int n_pixels);

void        gmm_run(GMMModel *model, const Image *img, const GMMConfig *cfg);

// log N(x|μ,Σ) = -½ [ d·log(2π) + log|Σ| + (x-μ)ᵀ Σ⁻¹ (x-μ) ]
float log_gaussian(const float *x, const GaussianComponent *comp, int n_bands)
{
    float quad = 0.0f;
    for (int i = 0; i < n_bands; i++) {
        float tmp = 0.0f;
        for (int j = 0; j < n_bands; j++) {
            float diff = x[j] - comp->mean[j];
            tmp += comp->covar_inv[i * n_bands + j] * diff;
        }
        quad += (x[i] - comp->mean[i]) * tmp;
    }
    float log_det = logf(comp->covar_det);
    return -0.5f * (n_bands * logf(2.0f * (float)M_PI) + log_det + quad);
}

// E STEP
// Calculamos la probabilidad a posteriori de que un punto x esté en un cluster k
double e_step(GMMModel *model, const Image *img)
{
    int k = model->k;
    int n_bands = model->n_bands;
    double log_likelihood = 0.0;
    float *log_probs = malloc(k * sizeof(float));

    for (int p = 0; p < img->n_pixels; p++) {
        const float *x = &img->data[(size_t)p * n_bands];
        if (is_nodata(x, img)) {
            for (int c = 0; c < k; c++)
                model->post[(size_t)p * k + c] = 0.0f;
            continue;
        }

        for (int c = 0; c < k; c++)
            log_probs[c] = logf(fmaxf(model->components[c].priori, FLT_MIN))
                    + log_gaussian(x, &model->components[c], n_bands);

        float lse_max = log_probs[0];
        for (int c = 1; c < k; c++)
            if (log_probs[c] > lse_max) lse_max = log_probs[c];

        float lse_sum = 0.0f;
        for (int c = 0; c < k; c++)
            lse_sum += expf(log_probs[c] - lse_max);
        double log_norm = (double)lse_max + logf(lse_sum);
        log_likelihood += log_norm;

        for (int c = 0; c < k; c++)
            model->post[(size_t)p * k + c] = expf(log_probs[c] - (float)log_norm);
    }

    free(log_probs);
    return log_likelihood;
}

// M STEP
// Actualizamos los parametros de las Gaussianas a partir de las posteriori calculadas en el step E
float m_step(GMMModel *model, const Image *img, double **new_means, double **new_covars, float **old_means)
{   
    int k = model->k;
    int n_bands = model->n_bands;
    int n_pixels = img->n_pixels;
    float max_shift = 0.0f;

    // DEBUG
    for (int c = 0; c < k; c++) {
        if (!isfinite(model->components[c].covar_det) || model->components[c].covar_det <= 0.0f)
            fprintf(stderr, "[M-step] component %d has bad det=%.6e before m_step\n", c, model->components[c].covar_det);
        float sum = 0.0f;
        for (int b = 0; b < model->n_bands; b++)
            sum += model->components[c].mean[b];
        if (!isfinite(sum))
            fprintf(stderr, "[M-step] component %d has NaN/Inf in mean\n", c);
    }

    // Guardamos medias antiguas
    for (int c = 0; c < k; c++) {
        memcpy(old_means[c], model->components[c].mean, n_bands * sizeof(float));
    }

    double *Nk = calloc(k, sizeof(double));
    for (int c = 0; c < k; c++) {
        memset(new_means[c],  0, n_bands * sizeof(double));
        memset(new_covars[c], 0, n_bands * n_bands * sizeof(double));
    }

    // Nuevas medias
    for (int p = 0; p < n_pixels; p++) {
        const float *x = &img->data[(size_t)p * n_bands];
        if (is_nodata(x, img)) continue;
        for (int c = 0; c < k; c++) {
            float r = model->post[(size_t)p * k + c];
            if (r == 0.0f) continue;
            Nk[c] += r;
            for (int i = 0; i < n_bands; i++)
                new_means[c][i] += (double)r * x[i];
        }
    }
    for (int c = 0; c < k; c++)
        if (Nk[c] > DBL_MIN)
            for (int i = 0; i < n_bands; i++)
                new_means[c][i] /= Nk[c];

    // Nuevas covarianzas
    for (int p = 0; p < n_pixels; p++) {
        const float *x = &img->data[(size_t)p * n_bands];
        if (is_nodata(x, img)) continue;
        for (int c = 0; c < k; c++) {
            float r = model->post[(size_t)p * k + c];
            if (r == 0.0f) continue;
            for (int i = 0; i < n_bands; i++) {
                double di = x[i] - new_means[c][i];
                for (int j = i; j < n_bands; j++) {
                    double dj = x[j] - new_means[c][j];
                    new_covars[c][i * n_bands + j] += r * di * dj;
                }
            }
        }
    }

    // Contamos pixeles validos
    int n_valid = 0;
    for (int p = 0; p < n_pixels; p++)
        if (!is_nodata(&img->data[(size_t)p * n_bands], img)) n_valid++;

    // Finalizamos medias, covarianzas y actualizamos componentes
    for (int c = 0; c < k; c++) {
        GaussianComponent *g = &model->components[c];

        // Componentes practicamente vacias -> les asignamos un pixel aleatorio
        if (Nk[c] < DBL_MIN) {
            int rp = get_random_pixel(img);

            for (int b = 0; b < n_bands; b++)
                model->components[c].mean[b] = img->data[rp * n_bands + b];
            memset(model->components[c].covar, 0, n_bands * n_bands * sizeof(float));

            for (int b = 0; b < n_bands; b++)
                model->components[c].covar[b * n_bands + b] = MIN_COVAR;
            model->components[c].priori = 1.0f / k;

            compute_covar_derived(g, n_bands);
            continue;
        }
        
        // Calculamos max. desplazamiento en medias
        for (int i = 0; i < n_bands; i++) {
            float new_val = (float)new_means[c][i];
            float d = fabsf(new_val - old_means[c][i]);
            if (d > max_shift) max_shift = d;
            g->mean[i] = new_val;
        }

        // Finalizamos covarianzas
        for (int i = 0; i < n_bands; i++) {
            for (int j = i; j < n_bands; j++) {
                float val = (float)(new_covars[c][i * n_bands + j] / Nk[c]);
                if (i == j) val += MIN_COVAR;
                new_covars[c][i * n_bands + j] = val;
                if (i != j) new_covars[c][j * n_bands + i] = val;
            }
        }

        // Calculamos max. desplazamiento en covarianzas
        for (int i = 0; i < n_bands; i++) {
            for (int j = 0; j < n_bands; j++) {
                float d = fabsf((float)new_covars[c][i * n_bands + j] - g->covar[i * n_bands + j]);
                if (d > max_shift) max_shift = d;
            }
        }

        // Actualizamos la componente
        g->priori = Nk[c] / n_valid;

        // Copiamos las covarianzas ya convertidas a float
        for (int i = 0; i < n_bands * n_bands; i++)
            g->covar[i] = (float)new_covars[c][i];

        compute_covar_derived(g, n_bands);
    }

    // Renormalizamos las a prioris
    float sum_priori = 0.0f;
    for (int c = 0; c < k; c++)
        sum_priori += model->components[c].priori;
    if (sum_priori > 0.0f) {
        for (int c = 0; c < k; c++)
            model->components[c].priori /= sum_priori;
    } else {
        for (int c = 0; c < k; c++)
            model->components[c].priori = 1.0f / k;
    }

    free(Nk);
    return max_shift;
}

// Obtenemos la mayor posteriori que define a que cluster pertenece cada pixel
int *get_hard_labels(const GMMModel *model, const Image *img, int n_pixels)
{
    int  k      = model->k;
    int *labels = malloc(n_pixels * sizeof(int));
    if (!labels) return NULL;

    for (int p = 0; p < n_pixels; p++) {
        if (is_nodata(&img->data[p * model->n_bands], img)) {
            labels[p] = -1;
            continue;
        }
        int   best = 0;
        float bval = model->post[(size_t)p * k];
        for (int c = 1; c < k; c++) {
            if (model->post[(size_t)p * k + c] > bval) {
                bval = model->post[(size_t)p * k + c];
                best = c;
            }
        }
        labels[p] = best;
    }
    return labels;
}

// Loop del algoritmo EM
void gmm_run(GMMModel *model, const Image *img, const GMMConfig *cfg)
{
    if (img->n_bands != model->n_bands) {
        fprintf(stderr, "[GMM]    Error: image has %d bands but model expects %d.\n",
                img->n_bands, model->n_bands);
        exit(EXIT_FAILURE);
    }

    int n_bands = model->n_bands;
    int k = model->k;

    model->post = calloc((size_t)img->n_pixels * k, sizeof(float));

    // Malloc de vectores de medias y matrices de covarianzas para m_step
    double **new_means = malloc(k * sizeof(double *));
    double **new_covars = malloc(k * sizeof(double *));
    float **old_means = malloc(k * sizeof(float *));
    for (int c = 0; c < k; c++) {
        new_means [c] = calloc(n_bands, sizeof(double));
        new_covars[c] = calloc(n_bands * n_bands, sizeof(double));
        old_means[c] = malloc(n_bands * sizeof(float));
    }

    double prev_ll = -DBL_MAX;
    int max_iter = cfg->max_iter;

    clock_t iter_start = clock();

    // En cada iteracion, hasta max_iter, realizamos un e_step (Expectation) y un m_step (Maximization)
    for (int iter = 0; iter < max_iter; iter++) {
        double ll = e_step(model, img);
        float shift = m_step(model, img, new_means, new_covars, old_means);

        double ll_change = (iter == 0) ? 0.0 : fabs((ll - prev_ll) / fmax(fabs(ll), 1.0));

        printf("[GMM]    iter %3d | log-likelihood = %.4f | ll delta = %.7f | max shift = %.6f\n",
               iter + 1, ll, ll_change, shift);

        // DEBUG
        if (iter > 0 && ll < prev_ll - 1.0) {
            fprintf(stderr, "[GMM]    WARNING: LL decreased by %.6f at iter %d - EM monotonicity violated.\n",
                    prev_ll - ll, iter + 1);
            for (int c = 0; c < k; c++) {
                fprintf(stderr, "         component %d: priori=%.6f  det=%.6e  mean=[", 
                        c, model->components[c].priori, model->components[c].covar_det);
                for (int b = 0; b < n_bands; b++)
                    fprintf(stderr, "%.2f%s", model->components[c].mean[b], b < n_bands-1 ? ", " : "");
                fprintf(stderr, "]\n");
            }

            double *post_sum = calloc(k, sizeof(double));
            memset(post_sum, 0, k * sizeof(double));
            for (int p = 0; p < img->n_pixels; p++)
                for (int c = 0; c < k; c++)
                    post_sum[c] += model->post[(size_t)p * k + c];
            fprintf(stderr, "         posterior sums: ");
            for (int c = 0; c < k; c++)
                fprintf(stderr, "[%d]=%.1f ", c, post_sum[c]);
            fprintf(stderr, "\n");
            free(post_sum);

            int bad_rows = 0;
            for (int p = 0; p < img->n_pixels; p++) {
                float row_sum = 0.0f;
                for (int c = 0; c < k; c++)
                    row_sum += model->post[(size_t)p * k + c];
                if (fabsf(row_sum - 1.0f) > 1e-3f)
                    bad_rows++;
            }
            if (bad_rows > 0)
                fprintf(stderr, "         WARNING: %d pixels have posterior rows not summing to 1.\n", bad_rows);
        }

        for (int c = 0; c < k; c++) {
            if (model->components[c].priori < (1.0f / k) * 0.01f)
                fprintf(stderr, "         component %d has suspiciously low priori (%.6f) - possible recent reinit.\n",
                        c, model->components[c].priori);
        }

        // Si el delta entre el log-likelihood es < cfg->convergence -> hemos acabado
        if (iter > 0 && ll_change < cfg->convergence) {
            printf("[GMM]    Converged at iteration %d.\n", iter + 1);
            e_step(model, img);

            model->labels = get_hard_labels(model, img, img->n_pixels);
            
            for (int c = 0; c < k; c++) { free(new_means[c]); free(new_covars[c]); free(old_means[c]); }
            free(new_means);
            free(new_covars);
            free(old_means);

            clock_t iter_end = clock();
            double iter_time = (double)(iter_end - iter_start) / CLOCKS_PER_SEC;
            printf("\n[Time]   EM loop elapsed: %.4f seconds\n", iter_time);
            return;
        }
        prev_ll = ll;
    }
    printf("[GMM]    Reached maximum iterations (%d).\n", max_iter);

    e_step(model, img);

    model->labels = get_hard_labels(model, img, img->n_pixels);

    clock_t iter_end = clock();
    double iter_time = (double)(iter_end - iter_start) / CLOCKS_PER_SEC;
    printf("\n[Time]   EM loop elapsed: %.4f seconds\n", iter_time);

    for (int c = 0; c < k; c++) { free(new_means[c]); free(new_covars[c]); free(old_means[c]); }
    free(new_means);
    free(new_covars);
    free(old_means);
}

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc < 3) {
        fprintf(stderr, "Usage:   %s <image_filepath> <k> [max_iter] [convergence] [seed] [kmeans_init_iter]\n", argv[0]);
        fprintf(stderr, "Example: %s sample.tif 4 100 0.003 12345 10\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];

    int k = atoi(argv[2]);
    if (k <= 0) {
        fprintf(stderr, "'k' must be a positive integer.\n");
        return 1;
    }
    if (k > MAX_K) {
        fprintf(stderr, "'k' cannot exceed %d (color palette / constant-memory limit).\n", MAX_K);
        return 1;
    }

    GMMConfig cfg;
    cfg.max_iter = 200;
    cfg.convergence = 1e-3f;
    unsigned int seed = 12345;
    lcg_set_seed(seed);
    cfg.kmeans_init_iter = 10;

    if (argc >= 4) {
        if(atoi(argv[3]) <= 0) {
            fprintf(stderr, "Invalid max_iterations: %d (must be > 0)\n", atoi(argv[3]));
            return 1;
        } else {
            cfg.max_iter = atoi(argv[3]);
        }
    }

    if (argc >= 5) {
        if((float)atof(argv[4]) <= 0) {
            fprintf(stderr, "Invalid convergence: %f (must be > 0)\n", (float)atof(argv[4]));
            return 1;
        } else {
            cfg.convergence = (float)atof(argv[4]);
        }
    }

    if (argc >= 6) {
        seed = (unsigned int)atoi(argv[5]);
        lcg_set_seed(seed);
    }

    if (argc >= 7) {
        if(atoi(argv[6]) <= 0) {
            fprintf(stderr, "Invalid k-means_init_iter: %d (must be > 0)\n", atoi(argv[6]));
            return 1;
        } else {
            cfg.kmeans_init_iter = atoi(argv[6]);
        }
    }

    GDALAllRegister();

    printf("=== GMM - CPU Version ===\n");
    printf("[Config] k=%d | image=\"%s\"\n", k, filename);
    printf("[Config] max_iter=%d | convergence=%.7f | seed=%u | k-means_init_iter=%d\n\n",
           cfg.max_iter, cfg.convergence, seed, cfg.kmeans_init_iter);

    Image *img = load_image(filename);
    if (!img) return 1;

    GMMModel *model = gmm_init(k, img->n_bands);

    clock_t init_start = clock();
    init_params_kmeans(model, img, cfg.kmeans_init_iter);
    clock_t init_end = clock();

    clock_t t_start = clock();
    gmm_run(model, img, &cfg);
    clock_t t_end   = clock();

    printf("[Time]   Total elapsed: %.4f seconds\n", (double)(t_end - t_start) / CLOCKS_PER_SEC);
    printf("[Time]   Initialization elapsed: %.4f seconds\n\n", (double)(init_end - init_start) / CLOCKS_PER_SEC);

    print_components(model);
    printf("\n");
    print_cluster_sizes(model->labels, img->n_pixels, k);
    printf("\n");
    save_csv(model->labels, img->n_pixels, "gmm_output.csv");
    save_image(model->labels, img->n_pixels, model->k, "gmm_output.tif", filename);

    gmm_free(model);
    free_image(img);
    return 0;
}