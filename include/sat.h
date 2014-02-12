#ifndef SAT_H
#define SAT_H
#include <gpudefs.h>
namespace sat {
typedef struct _alg_setup {
    int width, ///< Image width
        height, ///< Image height
        colGroupCount, ///< Number of column blocks/groups
        rowGroupCount, ///< Number of row blocks/groups
        iLastColGroup, ///< Last valid column-block
        iLastRowGroup, ///< Last valid row-block
        border, ///< Border extension to consider outside image
        carryHeight, ///< Auxiliary carry-image height
        carryWidth; ///< Auxiliary carry-image width
    float invWidth, ///< Inverse of image width
        invHeight; ///< Inverse of image height
} satConstants; ///< @see _alg_setup

void generateConstants( satConstants& algs,
                     const int& w,
                     const int& h );
void computeSummedAreaTable(float *h_inout,
        const int& w,
        const int& h);

} //end namespace sat
#endif
