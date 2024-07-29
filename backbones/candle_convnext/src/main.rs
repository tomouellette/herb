use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, LayerNorm, Conv2d, Module};
use candle_nn::{linear, layer_norm, conv2d, VarBuilder, Init};


#[derive(Debug)]
struct ConvNeXtLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl ConvNeXtLayerNorm {
    pub fn new(
        vb: &VarBuilder,
        dim: usize,
        prefix: &str,
    ) -> Result<Self> {
        let weight_key = format!("{}.weight", prefix).as_str().to_owned();
        let bias_key = format!("{}.bias", prefix).as_str().to_owned();
        let weight = vb.get_with_hints(dim, &weight_key, Init::Const(1.))?;
        let bias = vb.get_with_hints(dim, &bias_key, Init::Const(0.))?;

        Ok(Self { weight, bias })
    }
}

impl Module for ConvNeXtLayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let u = x.mean_keepdim(1)?;
        let s = x
            .broadcast_sub(&u)?
            .sqr()?
            .mean_keepdim(1)?;

        let x = x
            .broadcast_sub(&u)?
            .broadcast_div(&(s + 1e-6)?.sqrt()?)?;

        let weight = self.weight.unsqueeze(1)?.unsqueeze(1)?;
        let bias = self.bias.unsqueeze(1)?.unsqueeze(1)?;

        let x = x.broadcast_mul(&weight)?;
        let x = x.broadcast_add(&bias)?;

        Ok(x)
    }
}


pub fn move_channel(x: &Tensor, channel_last: bool) -> Result<Tensor> {
    if channel_last {
        Ok(x.permute([0, 2, 3, 1])?)
    } else {
        Ok(x.permute([0, 3, 1, 2])?)
    }
}

#[derive(Debug)]
struct ConvNeXtBlock {
    pub depthwise_convolution: Conv2d,
    pub norm: LayerNorm,
    pub linear1: Linear,
    pub gamma: Tensor,
    pub beta: Tensor,
    pub linear2: Linear,
}

impl ConvNeXtBlock {
    pub fn new(
        vb: &VarBuilder,
        in_chans: usize,
        stage: usize,
        block: usize
    ) -> Result<Self> {
        let prefix = format!("stages.{}.convnext_block{}", stage, block);

        let depthwise_config = candle_nn::Conv2dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: in_chans,
        };

        let depthwise_convolution = conv2d(
            in_chans,
            in_chans,
            7,
            depthwise_config,
            vb.pp(format!("{}.depthwise_convolution", prefix).as_str())
        )?;

        let prefix = format!("{}.pointwise_convolutions", prefix);

        let norm = layer_norm(in_chans, 1e-6, vb.pp(format!("{}.pw_norm", prefix).as_str()))?;
        let linear1 = linear(in_chans, in_chans * 4, vb.pp(format!("{}.pw_linear1", prefix).as_str()))?;

        let gamma = vb.get_with_hints(
            &[1, 1, 1, in_chans * 4],
            format!("{}.pw_grn.gamma", prefix).as_str(),
            Init::Const(0.))?;

        let beta = vb.get_with_hints(
            &[1, 1, 1, in_chans * 4],
            format!("{}.pw_grn.beta", prefix).as_str(),
            Init::Const(0.))?;

        let linear2 = linear(in_chans * 4, in_chans, vb.pp(format!("{}.pw_linear2", prefix).as_str()))?;

        Ok(Self {
            depthwise_convolution,
            norm,
            linear1,
            gamma,
            beta,
            linear2,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.depthwise_convolution.forward(x)?;
        let x = move_channel(&x, true)?;

        let x = self.norm.forward(&x)?;
        let x = self.linear1.forward(&x)?.gelu()?;

        let x_global = x
            .sqr()?
            .sum_keepdim([1, 2])?
            .mean_keepdim([1, 2])?
            .sqrt()?;

        let x_global_mean = x_global.mean_keepdim(3)?;
        let x_norm = x.broadcast_div(&(x_global_mean + 1e-6)?)?;
        let x_norm = x
            .broadcast_mul(&x_norm)?
            .broadcast_mul(&self.gamma)?
            .broadcast_add(&self.beta)?;

        let x = (x_norm + x)?;
        let x = self.linear2.forward(&x)?;

        let x = move_channel(&x, false)?;

        Ok(x)
    }
}

#[derive(Debug)]
struct ConvNeXtv2 {
    pub downsample_norm: Vec<ConvNeXtLayerNorm>,
    pub downsample_conv: Vec<Conv2d>,
    pub stages: Vec<Vec<ConvNeXtBlock>>,
    pub head_norm: LayerNorm,
    pub head: Linear,
}

impl ConvNeXtv2 {
    pub fn new(
        vb: &VarBuilder,
        in_chans: usize,
        n_classes: usize,
        depths: Vec<usize>,
        dims: Vec<usize>,
    ) -> Result<Self> {
        let in_conv_config = candle_nn::Conv2dConfig {
            padding: 0,
            stride: 4,
            dilation: 1,
            groups: 1,
        };

        let downsample_conv_config = candle_nn::Conv2dConfig {
            padding: 0,
            stride: 2,
            dilation: 1,
            groups: 1,
        };

        let mut downsample_conv = Vec::new();
        let mut downsample_norm = Vec::new();

        downsample_conv.push(conv2d(in_chans, dims[0], 4, in_conv_config, vb.pp("downsample_layers.0.in_conv"))?);
        downsample_norm.push(ConvNeXtLayerNorm::new(vb, dims[0], "downsample_layers.0.in_norm")?);

        for i in 0..3 {
            let prefix = format!("downsample_layers.{}.", i+1);
            let norm_key = format!("{}downsample_norm{}", prefix, i).as_str().to_owned();
            let conv_key = format!("{}downsample_conv{}", prefix, i).as_str().to_owned();

            let norm = ConvNeXtLayerNorm::new(&vb, dims[i], &norm_key)?;
            let conv = conv2d(dims[i], dims[i+1], 2, downsample_conv_config, vb.pp(conv_key))?;

            downsample_norm.push(norm);
            downsample_conv.push(conv);
        }
        
        let mut stages = Vec::new();
        for i in 0..depths.len() {
            let mut blocks: Vec<ConvNeXtBlock> = Vec::new();
            for j in 0..depths[i] {
                let block = ConvNeXtBlock::new(vb, dims[i], i, j)?;
                blocks.push(block);
            }
            stages.push(blocks);
        }

        let last_dim = dims.into_iter().last().unwrap();
        let head_norm = layer_norm(last_dim, 1e-6, vb.pp("norm"))?;
        let head = linear(last_dim, n_classes, vb.pp("head"))?;

        Ok(Self {
            downsample_norm,
            downsample_conv,
            stages,
            head_norm,
            head,
        })
    }

    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let mut weights = candle_core::safetensors::load(path, &device)?;

        let in_chans = weights.get("in_chans").unwrap().to_vec1::<i64>()?[0] as usize;
        let depths = weights
            .get("depths").unwrap()
            .to_vec1::<i64>()?
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();

        let dims = weights
            .get("dims").unwrap()
            .to_vec1::<i64>()?
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();

        let n_classes = if !weights.contains_key("head.weight") {
            0
        } else {
            weights.get("head.weight").unwrap().shape().dims()[0]
        };

        weights.remove("in_chans");
        weights.remove("depths");
        weights.remove("dims");
        weights.remove("out_dim");
        weights.remove("n_classes");

        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);

        Self::new(&vb, in_chans, n_classes, depths, dims)
    }
}

impl Module for ConvNeXtv2 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, stage) in self.stages.iter().enumerate() {
            if i == 0 {
                x = self.downsample_conv[i].forward(&x)?;
                x = self.downsample_norm[i].forward(&x)?;
            } else {
                x = self.downsample_norm[i].forward(&x)?;
                x = self.downsample_conv[i].forward(&x)?;
            }

            for block in stage {
                x = block.forward(&x)?;
            }
        }

        let x = x.mean([2, 3])?;
        let x = self.head_norm.forward(&x)?;
        let x = self.head.forward(&x)?;

        Ok(x)
    }
}


pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}
        
fn main() {
    let device = device(false).unwrap_or(Device::Cpu);

    let x = Tensor::rand(0.0_f32, 1.0_f32, &[1, 3, 224, 224], &device).unwrap();
    let x = x.to_dtype(candle_core::DType::F32).unwrap();
    let model = ConvNeXtv2::load("convnext.safetensors", &device).unwrap();
    let output = model.forward(&x).unwrap();

    println!("ConxNeXtv2 Output: {:?}", output.shape());
}
