use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use tokenizers::Tokenizer;
use crate::llm::token_output_stream::TokenOutputStream;
use tokio::sync::mpsc::{UnboundedSender};
use crate::llm::llm::TextGeneration;


use crate::llm::mistral_llm::mistral_initialization::{ Model};




impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {

        let logits_processor = LogitsProcessor::new(seed, temp, top_p);

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub(crate) fn run(&mut self, prompt: &str, sample_len: usize, tx:UnboundedSender<String>,context:&str) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();

        // Text Generation Prompt for Mistral
        let prompt=format!("<s>[INST]{}[/INST]",prompt.trim());

        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();


        let mut generated_tokens = 0usize;

        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };

        let start_gen = std::time::Instant::now();

        for index in 0..sample_len {

            let context_size = if index > 0 { 1 } else { tokens.len() };

            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];

            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::Quantized(m) => m.forward(&input, start_pos)?,
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                let _ = tx.send(t.to_string());
            }
        }


        let dt = start_gen.elapsed();

        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            let _ = tx.send(rest.to_string());
        }

        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        Ok(())
    }


}