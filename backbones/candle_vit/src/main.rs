use std::string::String;
use std::collections::HashMap;

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result, Tensor, IndexOp};
use candle_nn::{Linear, LayerNorm, Conv1d, Module};
use candle_nn::{linear, linear_no_bias, layer_norm, conv1d, VarBuilder, Init};
use candle_nn::ops::softmax;

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    norm: LayerNorm,
    to_out: Linear,
    dim_head: usize,
    heads: usize,
    scale: f64,
    project_out: bool,
}

impl Attention {
    fn new(
        vb: &VarBuilder,
        layer: usize,
        dim: usize,
        heads: usize,
        dim_head: usize,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let project_out = !(heads == 1 && dim_head == dim);

        let prefix = format!("transformer.attention_{}", layer);

        let norm = layer_norm(dim, 1e-5, vb.pp(format!("{}.norm", prefix).as_str()))?;
        let qkv = linear_no_bias(dim, inner_dim * 3, vb.pp(format!("{}.to_qkv", prefix).as_str()))?;
        let to_out = linear_no_bias(inner_dim, dim, vb.pp(format!("{}.to_out.0", prefix).as_str()))?;
        let scale = 1.0 / (dim_head as f64).sqrt();

        Ok(Self { qkv, norm, to_out, dim_head, heads, scale, project_out })
    }
}

impl Module for Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;

        let [q, k, v] = self.qkv
            .forward(&x)?
            .chunk(3, 2)?
            .try_into()
            .unwrap();

        let shape = q.shape().dims();
        let b = shape[0];
        let n = shape[1];
        let h = shape[2];
        let d = (h / self.heads) as usize;

        let q = q.reshape(&[b, n, self.heads, d])?;
        let q = q.permute([0, 2, 1, 3])?.contiguous()?;

        let k = k.reshape(&[b, n, self.heads, d])?;
        let k = k.permute([0, 2, 1, 3])?.contiguous()?;

        let v = v.reshape(&[b, n, self.heads, d])?;
        let v = v.permute([0, 2, 1, 3])?.contiguous()?;

        let attn = softmax(&(q.matmul(&k.transpose(3, 2)?)? * self.scale)?, 3)?;

        let out = attn.matmul(&v)?
            .permute([0, 2, 1, 3])?
            .reshape(&[b, n, self.heads * self.dim_head])?;

        if self.project_out {
            self.to_out.forward(&out)
        } else {
            Ok(out)
        }
    }
}

struct Feedforward {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl Feedforward {
    fn new(vb: &VarBuilder, layer: usize, dim: usize, expansion: usize) -> Result<Self> {
        let prefix = format!("transformer.feedforward_{}", layer);

        let norm = layer_norm(dim, 1e-5, vb.pp(format!("{}.norm", prefix).as_str()))?;
        let linear1 = linear(dim, expansion, vb.pp(format!("{}.linear1", prefix).as_str()))?;
        let linear2 = linear(expansion, dim, vb.pp(format!("{}.linear2", prefix).as_str()))?;

        Ok(Self { norm, linear1, linear2 })
    }
}

impl Module for Feedforward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.linear1.forward(&x)?.gelu()?;
        self.linear2.forward(&x)
    }
}

struct Transformer {
    norm: LayerNorm,
    attention: Vec<Attention>,
    feedforward: Vec<Feedforward>,
}

impl Transformer {
    fn new(
        vb: &VarBuilder,
        depth: usize,
        dim: usize,
        heads: usize,
        dim_head: usize,
        expansion: usize,
    ) -> Result<Self> {
        let norm = layer_norm(dim, 1e-5, vb.pp("transformer.norm"))?;

        let attention = (0..depth)
            .map(|layer| {
                Attention::new(vb, layer, dim, heads, dim_head)
            })
            .collect::<Result<Vec<_>>>()?;

        let feedforward = (0..depth)
            .map(|layer| {
                Feedforward::new(vb, layer, dim, expansion)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { norm, attention, feedforward })
    }
}

impl Module for Transformer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (attention, feedforward) in self.attention.iter().zip(&self.feedforward) {
            x = (&attention.forward(&x)? + x)?;
            x = (&feedforward.forward(&x)? + x)?;
        }
        self.norm.forward(&x)
    }
}

struct ViT {
    patch_height: usize,
    patch_width: usize,
    patch_embed_norm1: LayerNorm,
    patch_embed_norm2: LayerNorm,
    patch_embed: Linear,
    pos_embed: Tensor,
    cls_token: Tensor,
    n_registers: usize,
    register_tokens: Tensor,
    transformer: Transformer,
    mean_pool: bool,
    head: Linear,
    n_classes: usize,
}

impl ViT {
    fn new(
        vb: &VarBuilder,
        image_height: usize,
        image_width: usize,
        patch_height: usize,
        patch_width: usize,
        in_chans: usize,
        dim: usize,
        depth: usize,
        heads: usize,
        dim_head: usize,
        expansion: usize,
        n_registers: usize,
        mean_pool: bool,
        n_classes: usize,
    ) -> Result<Self> {
        if image_height % patch_height != 0 || image_width % patch_width != 0 {
            println!("Image dimensions must be divisible by the patch size");
            std::process::exit(1);
        }

        let n_patches = (image_height / patch_height) * (image_width / patch_width);
        let patch_dim = in_chans * patch_height * patch_width;

        let patch_embed_norm1 = layer_norm(patch_dim, 1e-5, vb.pp("to_patch_embedding.0"))?;
        let patch_embed = linear(patch_dim, dim, vb.pp("to_patch_embedding.1"))?;
        let patch_embed_norm2 = layer_norm(dim, 1e-5, vb.pp("to_patch_embedding.2"))?;
        
        let pos_embed = vb.get_with_hints(
            &[1, n_patches + 1, dim],
            "pos_embedding",
            Init::Const(0.))?;

        let cls_token = vb.get_with_hints(
            &[1, 1, dim],
            "cls_token",
            Init::Const(0.))?;

        let register_tokens = vb.get_with_hints(
            &[n_registers, dim],
            "register_tokens",
            Init::Const(0.))?;

        let transformer = Transformer::new(vb, depth, dim, heads, dim_head, expansion)?;

        let head = linear(dim, n_classes, vb.pp("head"))?;

        Ok(Self {
            patch_height,
            patch_width,
            patch_embed_norm1,
            patch_embed_norm2,
            patch_embed,
            pos_embed,
            cls_token,
            n_registers,
            register_tokens,
            transformer,
            mean_pool,
            head,
            n_classes,
        })
    }

    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let mut weights = candle_core::safetensors::load(path, &device)?;

        let image_height = weights.get("image_height").unwrap().to_vec1::<i64>()?[0] as usize;
        let image_width = weights.get("image_width").unwrap().to_vec1::<i64>()?[0] as usize;
        let patch_height = weights.get("patch_height").unwrap().to_vec1::<i64>()?[0] as usize;
        let patch_width = weights.get("patch_width").unwrap().to_vec1::<i64>()?[0] as usize;
        let in_chans = weights.get("in_chans").unwrap().to_vec1::<i64>()?[0] as usize;
        let dim = weights.get("dim").unwrap().to_vec1::<i64>()?[0] as usize;
        let depth = weights.get("depth").unwrap().to_vec1::<i64>()?[0] as usize;
        let heads = weights.get("heads").unwrap().to_vec1::<i64>()?[0] as usize;
        let mlp_dim = weights.get("mlp_dim").unwrap().to_vec1::<i64>()?[0] as usize;
        let dim_head = weights.get("dim_head").unwrap().to_vec1::<i64>()?[0] as usize;
        let n_registers = weights.get("n_registers").unwrap().to_vec1::<i64>()?[0] as usize;
        let mean_pool = weights.get("mean_pool").unwrap().to_vec1::<i64>()?[0] == 1;

        let n_classes = if !weights.contains_key("head.weight") {
            0
        } else {
            weights.get("head.weight").unwrap().shape().dims()[0]
        };

        for key in &[
            "image_height", "image_width",
            "patch_height", "patch_width",
            "in_chans", "dim", "depth",
            "heads", "mlp_dim", "dim_head", 
            "n_registers", "n_classes",
            "mean_pool"
        ] {
            weights.remove(*key);
        }

        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);

        Self::new(
            &vb,
            image_height,
            image_width,
            patch_height,
            patch_width,
            in_chans,
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            n_registers,
            mean_pool,
            n_classes,
        )
    }
}

impl Module for ViT {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims();
        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];

        let p1 = self.patch_height;
        let p2 = self.patch_width;

        let ph = h / p1;
        let pw = w / p2;

        let x = x.reshape(&[b, c, ph, p1, pw, p2])?;
        let x = x.permute([0, 2, 4, 3, 5, 1])?;
        let x = x.reshape(&[b, ph * pw, c * p1 * p2])?;


        let x = self.patch_embed_norm1.forward(&x)?;
        let x = self.patch_embed.forward(&x)?;
        let x = self.patch_embed_norm2.forward(&x)?;

        let cls_tokens = self.cls_token.expand(&[b, 1, x.shape().dims()[2]])?;
        let x = Tensor::cat(&[cls_tokens, x], 1)?;

        let x = (x + &self.pos_embed.i((.., ..(ph * pw + 1), ..))?)?;

        let r = self.register_tokens.repeat(&[b, 1, 1])?;

        let x = Tensor::cat(&[r, x], 1)?;
        let x = self.transformer.forward(&x)?;
        let x = x.i((.., self.n_registers.., ..))?;

        let x = if self.mean_pool {
            x.mean([1])?
        } else {
            x.i((.., 0..1))?
        };

        if self.n_classes > 0 {
            self.head.forward(&x)
        } else {
            Ok(x)
        }
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
    let model = ViT::load("vit.safetensors", &device).unwrap();
    let output = model.forward(&x).unwrap();

    println!("ViT Output: {:?}", output.shape());
}
