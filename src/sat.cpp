/*
 * sat.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: algomorph
 */
#include <sat.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <iostream>
#include <oclUtils.h>
#include <ClState.h>
#include <gpudefs.h>
#include <boost/lexical_cast.hpp>
namespace sat{

void generateConstants( satConstants& sc,
                     const int& w,
                     const int& h ) {

    sc.width = w;
    sc.height = h;
    sc.m_size = (w+WS-1)/WS;
    sc.n_size = (h+WS-1)/WS;
    sc.last_m = sc.m_size-1;
    sc.last_n = sc.n_size-1;
    sc.border = 0;
    sc.carry_width = sc.m_size*WS;
    sc.carry_height = sc.n_size*WS;
    sc.carry_height = h;
    sc.inv_width = 1.f/(float)w;
    sc.inv_height = 1.f/(float)h;

}

std::string
defineSymbol(const char* name,const std::string& value){
	return " -D " + std::string(name) + "=" + value;
}

std::string
genSatClDefineOptions( const satConstants& sc ) {
	return
	defineSymbol("WIDTH",boost::lexical_cast<std::string>(sc.width)) +
    defineSymbol("HEIGHT",boost::lexical_cast<std::string>(sc.height)) +
    defineSymbol("M_SIZE",boost::lexical_cast<std::string>(sc.m_size)) +
    defineSymbol("N_SIZE",boost::lexical_cast<std::string>(sc.n_size)) +
    defineSymbol("LAST_M",boost::lexical_cast<std::string>(sc.last_m)) +
    defineSymbol("LAST_N",boost::lexical_cast<std::string>(sc.last_n)) +
    defineSymbol("BORDER",boost::lexical_cast<std::string>(sc.border)) +
    defineSymbol("CARRY_WIDTH",boost::lexical_cast<std::string>(sc.carry_width)) +
    defineSymbol("CARRY_HEIGHT",boost::lexical_cast<std::string>(sc.carry_height)) +
    defineSymbol("INV_WIDTH",boost::lexical_cast<std::string>(sc.inv_width)) +
    defineSymbol("INV_HEIGHT",boost::lexical_cast<std::string>(sc.inv_height));
}

void prepare_algSAT( satConstants& sc,
                     float* d_inout,
                     float* d_ybar,
                     float* d_vhat,
                     float* d_ysum,
                     const float *h_in,
                     const int& w,
                     const int& h ) {

    sc.width = w;
    sc.height = h;

    if( w % 32 > 0 ) sc.width += (32 - (w % 32));
    if( h % 32 > 0 ) sc.height += (32 - (h % 32));

    generateConstants( sc, sc.width, sc.height );
    genSatClDefineOptions( sc );

    //d_inout.copy_from( h_in, w, h, algs.width, algs.height );
    //d_ybar.resize( algs.n_size * algs.width );
    //d_vhat.resize( algs.m_size * algs.height );
    //d_ysum.resize( algs.m_size * algs.n_size );

}

void algSAT( float* h_inout,
        const int& w,
        const int& h ) {
	satConstants algs;
	float* d_out, *d_ybar, *d_vhat, *d_ysum;
	prepare_algSAT( algs, d_out, d_ybar, d_vhat, d_ysum, h_inout, w, h );


}
} //end namespace sat

int main(int argc, char** argv){
	ClState state(true);
}
