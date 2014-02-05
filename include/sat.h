#ifndef SAT_H
#define SAT_H
#include <gpudefs.h>
namespace sat {
void algSAT(float* d_out, float* d_ybar, float* d_vhat, float* d_ysum,
		const float* d_in);
} //end namespace sat
#endif
