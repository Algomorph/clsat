#ifndef SAT_H
#define SAT_H
#include <gpudefs.h>
namespace sat {
typedef struct _alg_setup {
    int width, ///< Image width
        height, ///< Image height
        m_size, ///< Number of column-blocks
        n_size, ///< Number of row-blocks
        last_m, ///< Last valid column-block
        last_n, ///< Last valid row-block
        border, ///< Border extension to consider outside image
        carry_height, ///< Auxiliary carry-image height
        carry_width; ///< Auxiliary carry-image width
    float inv_width, ///< Inverse of image width
        inv_height; ///< Inverse of image height
} satConstants; ///< @see _alg_setup

void generateConstants( satConstants& algs,
                     const int& w,
                     const int& h );
void algSAT(float *h_inout,
        const int& w,
        const int& h);

} //end namespace sat
#endif
