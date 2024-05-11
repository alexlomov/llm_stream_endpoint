use std::str::FromStr;
use strum_macros::EnumString;

use twelf::{config, Layer};

#[config]
#[derive(Debug, Default)]
pub struct Config {
  number_of_workers: usize,
  llm_type: LlmType,
  mistral_config: Option<MistralConfig>
}

pub struct MistralConfig {
  acceleration: Acceleration,
  temperature: f64,
  use_flash_attention: bool,
  top_p: f64,
  seed: usize,
  sample_length: usize,
  model_id: String,
  model_file: String,
  tokenizer_id: String,
  tokenizer_file: String,
  weight_files: Option<String>,
  repeat_penalty: f64,
  repeat_last_n: usize
}

#[derive(Debug, PartialEq, EnumString)]
pub enum LlmType {
  Mistral, PhiV2, Llama
}

#[derive(Debug, PartialEq, EnumString)]
pub enum Acceleration {
  Cpu, Cuda, Metal, TryCuda, TryMetal
}

#[cfg(test)]
mod tests {
  use super::*;

  
  fn it_reads_cfg() {
    let cfg = Config::with_layers(
      &[Layer::Toml("./config/config.toml")]
    ).unwrap();

    assert_eq!(cfg.number_of_workers, 4);

  }

}


