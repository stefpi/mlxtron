# MLXtron (work in progress)

4D parallelizable training for models using MLX. Based on [Picotron](https://github.com/huggingface/picotron).

very minimal implementation and probably will only support LLama architecture for now.

as mac users we mostly operate in the GPU-poor case :sob:, but with enough macs together some real power kicks in.

read the blog to learn along with me at [stefpi.net/blog/](https://stefpi.net/blog)


## design

the benefit of training across multiple macs (aside from the biggest consumer RAM capacity) is the fact that each GPU used in the training network is attached to a significant amount of storage and CPU power. With this it gives us the option to skip many communication/broadcast steps because each device can have the dataset locally and pull only necessary samples into unified memory.