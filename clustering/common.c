#include "common.h"

// Generador de numeros pseudoaleatorios (LCG)
static unsigned int lcg_state;

void lcg_set_seed(unsigned int seed) {
    lcg_state = seed;
}

unsigned int lcg_rand(void) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return (lcg_state >> 16) & 0x7FFF;   // 15 bits
}

// Paleta Color Brewer : https://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
const int CLASS_COLORS[NUM_CLASS_COLORS][3] = {
    {31,120,180}, {51,160,44}, {227,26,28}, {255,127,0},
    {106,61,154}, {177,89,40}, {166,206,227}, {178,223,138},
    {251,154,153}, {253,191,111}, {202,178,214}, {255,255,153}
};

// GDAL I/O
Image *load_image(const char *filepath)
{
    GDALDatasetH dataset = GDALOpen(filepath, GA_ReadOnly);
    if (!dataset) {
        fprintf(stderr, "[GDAL] Error: cannot open '%s'\n", filepath);
        return NULL;
    }

    int width = GDALGetRasterXSize(dataset);
    int height = GDALGetRasterYSize(dataset);
    int bands = GDALGetRasterCount(dataset);

    printf("[GDAL]   Image loaded: %dx%d, %d bands\n", width, height, bands);

    Image *img = malloc(sizeof(Image));
    if (!img) { GDALClose(dataset); return NULL; }

    img->n_pixels = width * height;
    img->n_bands = bands;

    img->has_nodata = 0;
    img->nodata_value = 0.0f;

    GDALRasterBandH b1 = GDALGetRasterBand(dataset, 1);
    int bNoData;
    float ndv = (float)GDALGetRasterNoDataValue(b1, &bNoData);
    if (bNoData) {
        img->has_nodata = 1;
        img->nodata_value = ndv;
    }

    img->data = malloc((size_t)img->n_pixels * bands * sizeof(float));
    if (!img->data) { free(img); GDALClose(dataset); return NULL; }

    float *buffer = malloc((size_t)img->n_pixels * sizeof(float));
    if (!buffer) { free(img->data); free(img); GDALClose(dataset); return NULL; }

    for (int b = 0; b < bands; b++) {
        GDALRasterBandH band = GDALGetRasterBand(dataset, b + 1);
        CPLErr err = GDALRasterIO(band, GF_Read, 0, 0, width, height,
                buffer, width, height, GDT_Float32, 0, 0);
        if (err != CE_None) {
            fprintf(stderr, "[GDAL] Error reading band %d\n", b + 1);
            free(buffer); free(img->data); free(img); GDALClose(dataset);
            return NULL;
        }
        for (int i = 0; i < img->n_pixels; i++)
            img->data[(size_t)i * bands + b] = buffer[i];
    }

    free(buffer);
    GDALClose(dataset);
    return img;
}

void save_image(const int *labels, int n_pixels, int k, const char *output_path, 
    const char *reference_path)
{
    GDALDatasetH ref = GDALOpen(reference_path, GA_ReadOnly);
    if (!ref) {
        fprintf(stderr, "[GDAL] Error: cannot open reference '%s'\n", reference_path);
        return;
    }

    int width = GDALGetRasterXSize(ref);
    int height = GDALGetRasterYSize(ref);

    GDALDriverH driver = GDALGetDriverByName("GTiff");
    GDALDatasetH out = GDALCreate(driver, output_path, width, height, 1, GDT_Byte, NULL);
    if (!out) {
        fprintf(stderr, "[GDAL] Error: cannot create output image '%s'.\n", output_path);
        GDALClose(ref);
        return;
    }

    double geo[6];
    if (GDALGetGeoTransform(ref, geo) == CE_None)
        GDALSetGeoTransform(out, geo);
    const char *proj = GDALGetProjectionRef(ref);
    if (proj) GDALSetProjection(out, proj);
    GDALClose(ref);

    GDALRasterBandH band = GDALGetRasterBand(out, 1);
    GDALColorTableH ct = GDALCreateColorTable(GPI_RGB);

    for (int i = 0; i < k; i++) {
        GDALColorEntry e = { CLASS_COLORS[i][0], CLASS_COLORS[i][1], CLASS_COLORS[i][2], 255 };
        GDALSetColorEntry(ct, i, &e);
    }
    GDALColorEntry ndEntry = {0, 0, 0, 0};
    GDALSetColorEntry(ct, k, &ndEntry);

    GDALSetRasterColorTable(band, ct);
    GDALDestroyColorTable(ct);
    GDALSetRasterNoDataValue(band, (double)k);

    unsigned char *byte_labels = malloc((size_t)width * height);
    if (!byte_labels) { GDALClose(out); return; }
    for (int i = 0; i < width * height; i++)
        byte_labels[i] = (labels[i] == -1) ? (unsigned char)k : (unsigned char)labels[i];

    CPLErr err = GDALRasterIO(band, GF_Write, 0, 0, width, height,
            byte_labels, width, height, GDT_Byte, 0, 0);
    if (err != CE_None)
        fprintf(stderr, "[GDAL] Error writing output image.\n");

    free(byte_labels);
    GDALClose(out);
    printf("[Output] Segmented image saved to '%s'\n", output_path);
}

void free_image(Image *img)
{
    if (!img) return;
    free(img->data);
    free(img);
}

void save_csv(const int *labels, int n_pixels, const char *filepath)
{
    FILE *f = fopen(filepath, "w");
    if (!f) { fprintf(stderr, "Error: cannot open '%s'.\n", filepath); return; }
    fprintf(f, "pixel,cluster\n");
    for (int p = 0; p < n_pixels; p++)
        fprintf(f, "%d,%d\n", p, labels[p]);
    fclose(f);
    printf("[Output] Labels saved to '%s'.\n", filepath);
}

// Convierte pixel-mayor (np * nb) a band-mayor con stride alineado (nb * padded_np).
// padded_np se redondea a múltiplo de 32 para alinear cada banda a un limite de 128 bytes.
float *transpose_to_band_major(const Image *img, size_t *padded_pixels)
{
    size_t np = (size_t)img->n_pixels;
    size_t padded_np = ((np + 31) / 32) * 32;
    *padded_pixels = padded_np;

    float *out = (float*)malloc(padded_np * img->n_bands * sizeof(float));
    if (!out) return NULL;

    memset(out, 0, padded_np * img->n_bands * sizeof(float));

    for (int p = 0; p < img->n_pixels; p++)
        for (int b = 0; b < img->n_bands; b++)
            out[(size_t)b * padded_np + p] = img->data[(size_t)p * img->n_bands + b];

    return out;
}

// Obtiene un pixel aleatorio que no sea NoData mediante el LCG
int get_random_pixel(const Image *img)
{
    int rp;
    do {
        unsigned int rn = (lcg_rand() << 15) | lcg_rand();
        rp = (int)((unsigned long long)rn * img->n_pixels >> 30);
    } while (is_nodata(&img->data[rp * img->n_bands], img));
    return rp;
}

void print_cluster_sizes(const int *labels, int n_pixels, int k)
{
    int *counts = calloc(k, sizeof(int));
    int nodata_count = 0;
    for (int p = 0; p < n_pixels; p++) {
        if (labels[p] == -1) nodata_count++;
        else counts[labels[p]]++;
    }

    printf("[Stats] Cluster sizes:\n");
    for (int c = 0; c < k; c++)
        printf("        cluster %2d -> %d pixels (%.1f%%)\n",
               c, counts[c], 100.0f * counts[c] / n_pixels);
    printf("        nodata     -> %2d pixels (%.1f%%)\n",
           nodata_count, 100.0f * nodata_count / n_pixels);

    free(counts);
}

void print_centroids(const KMeansModel *model)
{
    printf("[Stats]  Final centroids:\n");
    for (int c = 0; c < model->k; c++) {
        printf("        cluster %2d: [", c);
        for (int b = 0; b < model->n_bands; b++) {
            printf("%.2f", model->centroids[c * model->n_bands + b]);
            if (b < model->n_bands - 1) printf(", ");
        }
        printf("]\n");
    }
}

void print_components(const GMMModel *model)
{
    printf("[Stats]  GMM components:\n");
    for (int c = 0; c < model->k; c++) {
        GaussianComponent *g = &model->components[c];
        printf("        component %2d (priori=%.4f): mean=[", c, g->priori);
        for (int b = 0; b < model->n_bands; b++) {
            printf("%.2f", g->mean[b]);
            if (b < model->n_bands - 1) printf(", ");
        }
        printf("]\n");
    }
}

// Init de centroides aleatorios
void init_centroids_random(KMeansModel *model, const Image *img)
{
    for (int c = 0; c < model->k; c++) {
        int rp = get_random_pixel(img);
        memcpy(&model->centroids[c * model->n_bands],
               &img->data[rp * img->n_bands],
               model->n_bands * sizeof(float));
    }
}

// Init de centroides k-means++
void init_centroids_pp(KMeansModel *model, const Image *img)
{
    int k = model->k;
    int nb = model->n_bands;
    float *centroids = model->centroids;

    // primer centroide
    int first = get_random_pixel(img);
    memcpy(centroids, &img->data[(size_t)first * nb], nb * sizeof(float));

    // distancias al centroide mas cercano
    float *dist = malloc(img->n_pixels * sizeof(float));
    for (int p = 0; p < img->n_pixels; p++)
        if (is_nodata(&img->data[p * nb], img))
            dist[p] = 0.0f;
        else
            dist[p] = euclidean_distance_sq(&img->data[p * nb], centroids, nb);

    // resto de centroides
    for (int c = 1; c < k; c++) {
        double sum = 0.0;
        for (int p = 0; p < img->n_pixels; p++) sum += dist[p];
            
        unsigned int rn = (lcg_rand() << 15) | lcg_rand();
        double r = (double)rn / 1073741824.0 * sum;
        int chosen = img->n_pixels - 1;

        double accum = 0.0;
        for (int p = 0; p < img->n_pixels; p++) {
            accum += dist[p];
            if (accum >= r) { chosen = p; break; }
        }

        while (is_nodata(&img->data[(size_t)chosen * nb], img)) {
            chosen = (chosen + 1) % img->n_pixels;
        }

        memcpy(&centroids[c * nb], &img->data[(size_t)chosen * nb], nb * sizeof(float));

        // actualizamos distancias
        const float *c_new = &centroids[c * nb];
        for (int p = 0; p < img->n_pixels; p++) {
            const float *pixel = &img->data[p * nb];
            float d = 0.0f;
            float best = dist[p];
            for (int b = 0; b < nb; b++) {
                float diff = pixel[b] - c_new[b];
                d += diff * diff;
                if (d >= best) break;
            }
            // actualizamos solo si el nuevo centroide esta mas cercano
            if (d < best)
                dist[p] = d;
        }
    }
    free(dist);
}

// init_iter iteraciones de k-means para alimentar GMM con centroides iniciales razonables
void init_params_kmeans(GMMModel *model, const Image *img, int init_iter)
{
    int k = model->k;
    int n_bands = model->n_bands;
    int n_pixels = img->n_pixels;

    float *centroids = (float*)calloc(k * n_bands, sizeof(float));
    float *new_c = (float*)calloc(k * n_bands, sizeof(float));
    int *labels = (int*)malloc(n_pixels * sizeof(int));
    int *counts = (int*)calloc(k, sizeof(int));

    // Centroides iniciales aleatorios
    for (int c = 0; c < k; c++) {
        int rp = get_random_pixel(img);
        memcpy(&centroids[c * n_bands], &img->data[rp * n_bands], n_bands * sizeof(float));
    }

    // Ejecutamos init_iter iteraciones
    for (int iter = 0; iter < init_iter; iter++) {
        memset(new_c,  0, k * n_bands * sizeof(float));
        memset(counts, 0, k * sizeof(int));

        // Calculamos distancia euclidea y asignamos cluster
        for (int p = 0; p < n_pixels; p++) {
            const float *pixel = &img->data[(size_t)p * n_bands];
            if (is_nodata(pixel, img)) {
                labels[p] = -1;
                continue;
            }
            float best = FLT_MAX;
            int   best_c = 0;
            for (int c = 0; c < k; c++) {
                float d = 0.0f;
                for (int b = 0; b < n_bands; b++) {
                    float diff = pixel[b] - centroids[c * n_bands + b];
                    d += diff * diff;
                }
                if (d < best) { best = d; best_c = c; }
            }
            labels[p] = best_c;
        }

        // Sumatorio de los pixeles de cada cluster
        for (int p = 0; p < n_pixels; p++) {
            int c = labels[p];
            if (c == -1) continue;
            counts[c]++;
            for (int b = 0; b < n_bands; b++)
                new_c[c * n_bands + b] += img->data[(size_t)p * n_bands + b];
        }

        // Calculamos nuevos centroides
        for (int c = 0; c < k; c++) {
            if (counts[c] == 0) continue;
            for (int b = 0; b < n_bands; b++)
                new_c[c * n_bands + b] /= counts[c];
        }
        memcpy(centroids, new_c, k * n_bands * sizeof(float));

        // Fix clusters vacios
        for (int c = 0; c < k; c++) {
            if (counts[c] == 0) {
                // Cojemos un pixel aleatorio de otro cluster y le asignamos el cluster vacio
                int rp = get_random_pixel(img);
                memcpy(&centroids[c * n_bands], &img->data[rp * n_bands], n_bands * sizeof(float));
            }
        }
    }

    // Utilizamos centroides finales de k-means para alimentar el modelo GMM
    int n_valid = 0;
    memset(counts, 0, k * sizeof(int));
    for (int p = 0; p < n_pixels; p++) {
        if (labels[p] != -1) {
            counts[labels[p]]++;
            n_valid++;
        }
    }

    for (int c = 0; c < k; c++) {
        GaussianComponent *g = &model->components[c];

        // Media
        memcpy(g->mean, &centroids[c * n_bands], n_bands * sizeof(float));

        // A priori
        if (n_valid > 0)
            g->priori = (counts[c] > 0) ? (float)counts[c] / n_valid : 1.0f / k;
        else
            g->priori = 1.0f / k;

        // Matriz de covarianzas
        double *covar_acc = (double*)calloc(n_bands * n_bands, sizeof(double));

        for (int p = 0; p < n_pixels; p++) {
            if (labels[p] != c) continue;
            for (int i = 0; i < n_bands; i++) {
                double di = (double)img->data[(size_t)p * n_bands + i] - g->mean[i];
                for (int j = i; j < n_bands; j++) {
                    double dj = (double)img->data[(size_t)p * n_bands + j] - g->mean[j];
                    covar_acc[i * n_bands + j] += di * dj;
                }
            }
        }

        // Espejo del triangulo superior
        for (int i = 0; i < n_bands; i++) {
            for (int j = 0; j < i; j++) {
                covar_acc[i * n_bands + j] = covar_acc[j * n_bands + i];
            }
        }

        int n = (counts[c] > 1) ? counts[c] : 1;
        for (int i = 0; i < n_bands; i++) {
            for (int j = 0; j < n_bands; j++)
                g->covar[i * n_bands + j] = (float)(covar_acc[i * n_bands + j] / n);
            g->covar[i * n_bands + i] += MIN_COVAR;
        }

        free(covar_acc);

        // Matriz de covarianzas inversa + determinante de la matriz de covarianzas
        compute_covar_derived(g, n_bands);
    }

    // Renormalizamos prioris
    float sum_priori = 0.0f;
    for (int c = 0; c < k; c++) sum_priori += model->components[c].priori;
    if (sum_priori > 0.0f) {
        for (int c = 0; c < k; c++)
            model->components[c].priori /= sum_priori;
    } else {
        for (int c = 0; c < k; c++)
            model->components[c].priori = 1.0f / k;
    }

    free(centroids);
    free(labels);
    free(counts);
    free(new_c);

    printf("[GMM]    Parameters seeded from %d-iteration random k-means.\n", init_iter);
}

// Calculamos la distancia euclidea entre vector a y vector b
float euclidean_distance_sq(const float *a, const float *b, int n)
{
    float dist = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// Descomposicion de Cholesky, A = L * L^T.
// Almacena L en el triangulo inferior de la matriz de salida.
// Devuelve 0 si descompone, -1 si A no es una matriz positiva definida.
int cholesky_decompose(const float *A, float *L, int n)
{
    // Copia triangulo inferior de A en L
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++)
            L[i * n + j] = A[i * n + j];
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = L[i * n + j];
            for (int k = 0; k < j; k++)
                sum -= L[i * n + k] * L[j * n + k];

            if (i == j) {
                if (sum <= 0.0f)
                    return -1; // No es positiva definida
                L[i * n + i] = sqrtf(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }
    return 0;
}

// Invierte A mediante L por sustitucion forward y backward.
void cholesky_invert(const float *L, float *A_inv, int n)
{
    float *b = malloc(n * sizeof(float));

    for (int col = 0; col < n; col++) {
        for (int i = 0; i < n; i++)
            b[i] = (i == col) ? 1.0f : 0.0f;

        // Sustitucion forward: L y = b
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++)
                b[i] -= L[i * n + j] * b[j];
            b[i] /= L[i * n + i];
        }

        // Sustitucion backward: L^T x = y
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++)
                b[i] -= L[j * n + i] * b[j];
            b[i] /= L[i * n + i];
        }

        for (int i = 0; i < n; i++)
            A_inv[i * n + col] = b[i];
    }
    free(b);
}

// Determinante de A a partir de su factor de Cholesky L.
// det(A) = det(L)^2 = (prod L[i*n+i])^2
float cholesky_determinant(const float *L, int n)
{
    float det = 1.0f;
    for (int i = 0; i < n; i++)
        det *= L[i * n + i];
    return det * det;
}

// Calculamos la matriz de covarianzas inversa y el determinante por cada cluster
void compute_covar_derived(GaussianComponent *g, int n_bands)
{
    float *L = malloc(n_bands * n_bands * sizeof(float));

    if (cholesky_decompose(g->covar, L, n_bands) == 0) {
        // 0 == matriz positiva definida
        cholesky_invert(L, g->covar_inv, n_bands);
        g->covar_det = cholesky_determinant(L, n_bands);

        // si determinante <= 0, lo ponemos a FLT_MIN (menor valor positivo y != 0 posible en float)
        if (g->covar_det <= 0.0f)
            g->covar_det = FLT_MIN;
    } else {
        // Warning (no deberia producirse)
        fprintf(stderr, "[GMM] WARNING: non-positive definite covariance, regularising.\n");

        // Regularizamos
        for (int b = 0; b < n_bands; b++)
            g->covar[b * n_bands + b] += MIN_COVAR * 10.0f;

        if (cholesky_decompose(g->covar, L, n_bands) == 0) {
            cholesky_invert(L, g->covar_inv, n_bands);
            g->covar_det = cholesky_determinant(L, n_bands);
            if (g->covar_det <= 0.0f)
                g->covar_det = FLT_MIN;
        } else {
            // Doble warning (si pasa es que hay un problema...)
            fprintf(stderr, "[GMM] WARNING: regularisation failed, resetting to identity.\n");
            memset(g->covar,     0, n_bands * n_bands * sizeof(float));
            memset(g->covar_inv, 0, n_bands * n_bands * sizeof(float));
            // Identidad regularizada
            for (int b = 0; b < n_bands; b++) {
                g->covar    [b * n_bands + b] = MIN_COVAR;
                g->covar_inv[b * n_bands + b] = 1.0f / MIN_COVAR;
            }
            g->covar_det = powf(MIN_COVAR, n_bands);
        }
    }
    free(L);
}

// Init del modelo
KMeansModel *kmeans_init(int k, int n_bands)
{
    KMeansModel *m = malloc(sizeof(KMeansModel));
    if (!m) return NULL;

    m->k = k;
    m->n_bands = n_bands;
    m->labels = NULL;

    m->centroids = calloc(k * n_bands, sizeof(float));
    if (!m->centroids) { free(m); return NULL; }
    
    return m;
}

// Free del modelo
void kmeans_free(KMeansModel *model)
{
    if (!model) return;

    free(model->centroids);
    free(model->labels);
    free(model);
}

// Init del modelo con init de cada gausiana (una por cluster)
GMMModel *gmm_init(int k, int n_bands)
{
    GMMModel *m = malloc(sizeof(GMMModel));
    if (!m) return NULL;

    m->k = k;
    m->n_bands = n_bands;
    m->post = NULL;
    m->labels = NULL;

    m->components = malloc(k * sizeof(GaussianComponent));
    if (!m->components) { free(m); return NULL; }

    for (int c = 0; c < k; c++) {
        GaussianComponent *g = &m->components[c];

        g->mean = calloc(n_bands, sizeof(float));
        g->covar = calloc(n_bands * n_bands, sizeof(float));
        g->covar_inv = calloc(n_bands * n_bands, sizeof(float));
        if (!g->mean || !g->covar || !g->covar_inv) {
            for (int j = 0; j <= c; j++) {
                free(m->components[j].mean);
                free(m->components[j].covar);
                free(m->components[j].covar_inv);
            }
            free(m->components);
            free(m);
            return NULL;
        }

        g->covar_det = 1.0f;
        g->priori    = 1.0f / k;
    }

    return m;
}

// Free del modelo
void gmm_free(GMMModel *model)
{
    if (!model) return;

    for (int c = 0; c < model->k; c++) {
        free(model->components[c].mean);
        free(model->components[c].covar);
        free(model->components[c].covar_inv);
    }
    free(model->components);
    free(model->post);
    free(model->labels);
    free(model);
}