# cog-flux

This is a [Cog](https://cog.run) inference model for FLUX.1 [schnell] and FLUX.1 [dev] by [Black Forest Labs](https://blackforestlabs.ai/). It powers the following Replicate models:

* https://replicate.com/black-forest-labs/flux-schnell
* https://replicate.com/black-forest-labs/flux-dev

## Features

* Compilation with `torch.compile`
* Optional fp8 quantization based on [aredden/flux-fp8-api](https://github.com/aredden/flux-fp8-api), using fast CuDNN attention from Pytorch nightlies
* NSFW checking with [CompVis](https://huggingface.co/CompVis/stable-diffusion-safety-checker) and [Falcons.ai](https://huggingface.co/Falconsai/nsfw_image_detection) safety checkers
* img2img support

## Getting started

If you just want to use the models, you can run [FLUX.1 [schnell]](https://replicate.com/black-forest-labs/flux-schnell) and [FLUX.1 [dev]](https://replicate.com/black-forest-labs/flux-dev) on Replicate with an API or in the browser.

The code in this repo can be used as a template for customizations on FLUX.1, or to run the models on your own hardware.

First you need to select which model to run:

```shell
script/select.sh {dev,schnell}
```

Then you can run a single prediction on the model using:

```shell
cog predict -i prompt="a cat in a hat"
```

The [Cog getting started guide](https://cog.run/getting-started/) explains what Cog is and how it works.

To deploy it to Replicate, run:

```shell
cog login
cog push r8.im/<your-username>/<your-model-name>
```

Learn more on [the deploy a custom model guide in the Replicate documentation](https://replicate.com/docs/guides/deploy-a-custom-model).

## Contributing

Pull requests and issues are welcome! If you see a novel technique or feature you think will make FLUX.1 inference better or faster, let us know and we'll do our best to integrate it.

## Rough, partial roadmap

* Serialize quantized model instead of quantizing on the fly
* Use row-wise quantization
* Port quantization and compilation code over to https://github.com/replicate/flux-fine-tuner

# Usage

## Installation

Install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc
source /root/.bashrc
```

To install pget (efficient model weight downloading tool from replicate )

```
curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" && chmod +x /usr/local/bin/pget
```


## For jupyter notebook

```
uv add ipykernel
uv run ipython kernel install --user --name=uv_test
```

# Inference

```
uv run python predict.py
```