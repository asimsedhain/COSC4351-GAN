FROM tacc/tacc-ml:ubuntu16.04-cuda10-tf2.1-pt1.3 

########################################
# Install mpi
########################################

# necessities and IB stack
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
		curl vim-nox gfortran bison libibverbs-dev libibmad-dev libibumad-dev librdmacm-dev libmlx5-dev libmlx4-dev && \
	docker-clean

		# Install mvapich2-2.2
ARG MAJV=2
ARG MINV=2
ARG DIR=mvapich${MAJV}-${MAJV}.${MINV}

RUN curl http://mvapich.cse.ohio-state.edu/download/mvapich/mv${MAJV}/${DIR}.tar.gz | tar -xzf - \
	&& cd ${DIR} \
	&& ./configure \
	&& make -j $(nproc --all 2>/dev/null || echo 2) && make install \
	&& mpicc examples/hellow.c -o /usr/bin/hellow \
	&& cd ../ && rm -rf ${DIR}

RUN HOROVOD_CUDA_HOME=/opt/apps/cuda/10.0 HOROVOD_GPU_ALLREDUCE=NCCL \
	HOROVOD_NCCL_HOME=/opt/apps/cuda10_0/nccl/2.4.7 HOROVOD_WITH_TENSORFLOW=1 \
	HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1
									
RUN pip uninstall tensorflow-gpu

RUN pip install tensorflow==2.1
RUN pip install horovod==0.16.4
