#!/bin/bash

INSTALL_IPOPT=false
CREATE_VENV=false

PREPARE_VENV=true
TEST_CVIKS=true

#
#  These variables are machine dependent
#

MOTHERDIR=/g100/home/userexternal/fmarino0

load_python_libs()
{
    module load python/3.11.7--gcc--10.2.0
    module load mkl
}

prepare_venv_func() {
    load_python_libs
    source ~/Ipopt_Venv/bin/activate
}

CVIKSDIR=/g100/home/userexternal/fmarino0/CV_IKS



# From now on, automatic
export MOTHERDIR=${MOTHERDIR}
export CVIKSDIR=${CVIKSDIR}
export IPOPT_DIR=${MOTHERDIR}/Ipopt        # Choose where to install Ipopt
export IPOPT_LIBDIR=${IPOPT_DIR}/lib

# MKL
export INCLUDE=$INCLUDE:$MKLROOT/include
export LIBRARY_PATH=$LIBRARY_PATH:$MKLROOT/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MKLROOT/lib/intel64

export BLAS_FLAGS="-L${MKLROOT}/lib/intel64 -lmkl_rt"
export LAPACK_FLAGS="-L${MKLROOT}/lib/intel64 -lmkl_rt"

# IPOPT
export LIBRARY_PATH=$LIBRARY_PATH:$IPOPT_LIBDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IPOPT_LIBDIR
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$IPOPT_LIBDIR/pkgconfig

echo "IPOPT_LIBDIR : ${IPOPT_LIBDIR}"
echo "PKG_CONFIG_PATH : $IPOPT_LIBDIR/pkgconfig"

if [ $INSTALL_IPOPT = true ]; then
    echo "Installing Ipopt"
    git clone https://github.com/coin-or/Ipopt.git
    cd Ipopt

    ./configure --prefix=${IPOPT_DIR} --with-blas="${BLAS_FLAGS}" --with-lapack="${LAPACK_FLAGS}"
    make
    make test
    make install

fi

if [ $CREATE_VENV = true ]; then

    # Check Ipopt is linked against MKL
    echo "Checking MKL"
    ldd ${IPOPT_LIBDIR}/libipopt.so | grep mkl
    
    echo ""
    echo "Creating virtual environment"
    load_python_libs

    cd $MOTHERDIR
    python3.11 -m venv Ipopt_Venv
    conda create --name Ipopt_Venv python=3.11
    source ~/Ipopt_Venv/bin/activate

    pkg-config --modversion ipopt

    pip3 install --upgrade pip
    pip3 install numpy==1.25.2
    pip3 install scipy matplotlib cython setuptools
    pip3 install --upgrade findiff

    export LDFLAGS="-L${MKLROOT}/lib"
    export CFLAGS="-I${MKLROOT}/include"
    pip3  install cyipopt --verbose

    # Test
    source ~/Ipopt_Venv/bin/activate
    cd "${CVIKSDIR}/Examples"
    echo "Testing Ipopt"
    python3 minimal.py
fi

# Execute as source Script_Ipopt.sh
if [ $PREPARE_VENV = true ]; then
    prepare_venv_func
fi

if [ $TEST_CVIKS = true ]; then
    echo "Testing CV_IKS"
    prepare_venv_func
    cd "${CVIKSDIR}/Examples"
    python3 example.py
fi