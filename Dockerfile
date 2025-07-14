FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && \
    apt-get install -y wget curl gnupg lsb-release && \
    # # Eliminar repositorios antiguos de CUDA y ML que causan errores
    # rm -f /etc/apt/sources.list.d/cuda.list && \
    # rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
    # Quitar la clave antigua
    apt-key del 7fa2af80 || true && \
    # Importar la nueva clave GPG para CUDA 10
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    # Importar la clave para Machine Learning si la necesitas
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y git bzip2

# Install other dependencies before conda
RUN apt-get update && apt-get install -y \
    wget curl gnupg lsb-release \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1


# Asegurarse de que wget esté presente antes de descargar Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p $CONDA_DIR && \
    rm /miniconda.sh && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc


# Crear entorno Conda
RUN conda create -n colon_superpoint python=3.6 -y && \
    conda install -n colon_superpoint -c conda-forge cudatoolkit=10.0 cudnn=7.6 -y && \
    conda install -n colon_superpoint -c anaconda tensorflow-gpu=1.13 -y

# Copiar tu código
WORKDIR /ColonSuperPoint
COPY . /ColonSuperPoint

# Install Superpoint and dependencies
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate colon_superpoint && \
    pip install -r requirements.txt && \
    pip install -e ."

# Set up paths automatically (instead of interactive setup.sh)
RUN echo "DATA_PATH = './ColonExperiments/datasets'" > ./superpoint/settings.py && \
    echo "EXPER_PATH = './ColonExperiments'" >> ./superpoint/settings.py

# Create directories for mounting
RUN mkdir -p /ColonSuperPoint/ColonExperiments/datasets

# Activate conda environment in the Dockerfile
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate colon_superpoint" >> ~/.bashrc

# Comando por defecto: activar entorno
SHELL ["/bin/bash", "-c"]
CMD ["source /opt/conda/etc/profile.d/conda.sh && conda activate colon_superpoint && bash"]
