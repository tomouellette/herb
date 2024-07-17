use std::string::String;
use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarMap, Linear, LayerNorm, Conv1d, Module};
use candle_nn::{linear, layer_norm, conv1d, VarBuilder};
use candle_nn::ops::dropout;

#[derive(Debug)]
struct MixerLayer {
    pub norm1: LayerNorm,
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub norm2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
    pub dropout_rate: f32,
}

impl MixerLayer {
    fn new(
        vb: &VarBuilder,
        layer: usize,
        dim: usize,
        num_patches: usize,
        expansion_factor: usize,
        expansion_factor_tokens: f32,
        dropout_rate: f32,
    ) -> Result<Self> {
        let config = candle_nn::Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 0,
            groups: 1,
        };

        let prefix = format!("mixer_layers.{}", layer);

        let channels_expansion = num_patches * expansion_factor;
        let norm1 = layer_norm(dim, 1e-5, vb.pp(format!("{}.norm1", prefix).as_str()))?;

        let conv1 = conv1d(
            num_patches, channels_expansion, 1,
            config,
            vb.pp(format!("{}.conv1", prefix).as_str())
        )?;

        let conv2 = conv1d(
            channels_expansion, num_patches, 1,
            config, vb.pp(format!("{}.conv2", prefix).as_str())
        )?;

        let token_expansion = (dim as f32 * expansion_factor_tokens) as usize;
        let norm2 = layer_norm(dim, 1e-5, vb.pp(format!("{}.norm2", prefix).as_str()))?;
        let fc1 = linear(dim, token_expansion, vb.pp(format!("{}.fc1", prefix).as_str()))?;
        let fc2 = linear(token_expansion, dim, vb.pp(format!("{}.fc2", prefix).as_str()))?;
        
        Ok(Self {
            norm1,
            conv1,
            conv2,
            norm2,
            fc1,
            fc2,
            dropout_rate,
        })
    }
}

impl Module for MixerLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm1.forward(x)?;
        let x = self.conv1.forward(&x)?.gelu()?;
        let x = dropout(&x, self.dropout_rate)?;
        let x = self.conv2.forward(&x)?;
        let x = dropout(&x, self.dropout_rate)?;
        let x = self.norm2.forward(&x)?;
        let x = self.fc1.forward(&x)?.gelu()?;
        let x = dropout(&x, self.dropout_rate)?;
        let x = self.fc2.forward(&x)?;
        dropout(&x, self.dropout_rate)
    }
}

#[derive(Debug)]
struct MixerHead {
    norm: LayerNorm,
    fc: Linear,
    num_patches: usize,
}

impl MixerHead {
    fn new(vb: &VarBuilder, dim: usize, num_patches: usize, n_classes: usize) -> Result<Self> {
        let norm = layer_norm(dim, 1e-5, vb.pp("head.0"))?;
        let fc = linear(dim, n_classes, vb.pp("head.2"))?;
        Ok(Self { norm, fc, num_patches })
    }
}

impl Module for MixerHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = x.unsqueeze(1)?;
        let x = x.avg_pool2d((self.num_patches, 1))?;
        let x = x.squeeze(1)?;
        self.fc.forward(&x)
    }
}

#[derive(Debug)]
struct MLPMixer {
    pub patch_embed: Linear,
    pub mixer_layers: Vec<MixerLayer>,
    pub head: MixerHead,
    pub with_head: bool,
    pub patch_size: usize,
}

impl MLPMixer {
    fn new(
        vb: VarBuilder,
        image_size: (usize, usize),
        in_chans: usize,
        patch_size: usize,
        dim: usize,
        depth: usize,
        n_classes: usize,
        expansion_factor: usize,
        expansion_factor_tokens: f32,
        dropout_rate: f32,
    ) -> Result<Self> {
        let (h, w) = image_size;

        if h % patch_size != 0 || w % patch_size != 0 {
            println!("Image dimensions must be divisible by the patch size");
            std::process::exit(1);
        }

        let num_patches = (h / patch_size) * (w / patch_size);

        println!("Number of patches: {}", num_patches);
        println!("Patch size: {}", patch_size);
        println!("Number of classes: {}", n_classes);
        println!("Input channels: {}", in_chans);
        println!("Embedding dimension: {}", dim);
        println!("Number of layers: {}", depth);
        println!("Expansion factor: {}", expansion_factor);
        println!("Expansion factor tokens: {}", expansion_factor_tokens);
        
        let patch_embed = linear(in_chans * patch_size * patch_size, dim, vb.pp("patch_embed"))?;

        let mixer_layers = (0..depth)
            .map(|layer| {
                MixerLayer::new(
                    &vb,
                    layer,
                    dim,
                    num_patches,
                    expansion_factor,
                    expansion_factor_tokens,
                    dropout_rate,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let (head, with_head) = if n_classes > 0 {
            (MixerHead::new(&vb, dim, num_patches, n_classes)?, true)
        } else {
            (MixerHead::new(&vb, dim, num_patches, 1)?, false)
        };
        
        Ok(Self {
            patch_embed,
            mixer_layers,
            head,
            with_head,
            patch_size,
        })
    }
}

impl Module for MLPMixer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims();

        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];

        let ps = self.patch_size;
        let ph = h / ps;
        let pw = w / ps;

        let x = x.reshape((b, c, ph, ps, pw, ps))?;
        let x = x.permute((0, 2, 4, 3, 5, 1))?;
        let x = x.reshape((b, ph * pw, c * ps * ps))?;

        let x = self.patch_embed.forward(&x)?;

        let mut x = x;
        for layer in &self.mixer_layers {
            x = layer.forward(&x)?;
        }

        if self.with_head {
            return self.head.forward(&x)?.squeeze(1);
        } else {
            return Ok(x);
        }
    }
}

        
fn main() {
    let device = Device::Cpu;
    let x = Tensor::rand(0.0, 1.0, &[1, 3, 224, 224], &device).unwrap();
    let x = x.to_dtype(candle_core::DType::F32).unwrap();

    let mut weights = candle_core::safetensors::load("mlp_mixer.safetensors", &device).unwrap();

    for (name, tensor) in weights.iter() {
        println!("{}: {:?}", name, tensor.shape());
    }

    let dim = weights
        .get("patch_embed.weight")
        .unwrap()
        .shape().dims()[0];

    let depth = weights
        .iter()
        .filter(|(name, _)| name.starts_with("mixer_layers"))
        .count() / 6 / 2;

    let parameters = weights
        .get("parameters")
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let image_size = (parameters[0] as usize, parameters[1] as usize);
    let in_chans = parameters[2] as usize;
    let patch_size = parameters[3] as usize;
    let expansion_factor = parameters[4] as usize;
    let expansion_factor_tokens = parameters[5];
    let dropout_rate = parameters[6];
    
    println!("weights patch_embed: {:?}", weights.get("patch_embed.weight").unwrap().shape());
    println!("bias patch_embed: {:?}", weights.get("patch_embed.bias").unwrap().shape());
    
    let n_classes = if !weights.contains_key("head.0.weight") {
        println!("head.fc.weight not detected. Only initializing embedding backbone.");
        0
    } else {
        weights.get("head.2.weight").unwrap().shape().dims()[0]
    };

    let mut hashmap: HashMap<String, Tensor> = HashMap::new();
    hashmap.insert("patch_embed".to_string(), weights.get("patch_embed.weight").unwrap().clone());

    // remove parameters from weights
    weights.remove("parameters");

    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    //  get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Tensor>
    // get patche embed
    // let patch_embed = vb.get((in_chans * patch_size * patch_size, dim), "patch_embed.weight").unwrap();
    // println!("patch_embedASDASDASDAS: {:?}", patch_embed.shape());

    let model = MLPMixer::new(
        vb,
        image_size,
        in_chans,
        patch_size,
        dim,
        depth,
        n_classes,
        expansion_factor,
        expansion_factor_tokens,
        dropout_rate,
    ).unwrap();

    let output = model.forward(&x).unwrap();
    println!("output: {:?}", output.shape());
}
