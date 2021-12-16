FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt update -y
RUN apt install -y task-spooler vim
RUN git clone https://github.com/openai/CLIP
RUN git clone https://github.com/CompVis/taming-transformers
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ftfy regex tqdm omegaconf pytorch-lightning Mastodon.py imageio pandas seaborn kornia einops==0.3.0 transformers==4.3.1
RUN git clone https://github.com/vgan/vqganclip.git
RUN chmod u+x vqganclip/start_jupyter_notebook.sh vqganclip/enqueue_generate_images.sh vqganclip/aiartbot.py
RUN mkdir -p "/root/.cache/torch/hub/checkpoints"
RUN curl "https://download.pytorch.org/models/vgg16-397923af.pth" -o "/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
RUN ln -s /mnt/vqganclip/models /tf/
RUN ln -s /mnt/vqganclip/outputs /tf/
WORKDIR "/tf/vqganclip"
ADD VERSION .
