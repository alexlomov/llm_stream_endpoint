#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::convert::Infallible;
use warp::{Filter};

use tokio_stream::wrappers::{UnboundedReceiverStream};
use tokio::sync::mpsc;
use tokio::sync::mpsc::{ UnboundedReceiver, UnboundedSender};

use std::{fs, thread};
use std::sync::{Arc, Mutex};

use bytes::{Bytes};

use futures_util::{Stream, StreamExt};
use hyper::Body;
use serde::{Deserialize, Serialize};

use llm_stream::args_init::args::Args;
use llm_stream::llm::llm::{LLM, LlmPackage,generate};

// todo: to be put under feature
use llm_stream::llm::mistral_llm::mistral_initialization::{ MistralLlm};


const NB_WORKERS:usize = 4;

#[derive(Serialize, Deserialize, Debug,Clone)]
pub struct Prompt {
    pub query: String,
}


#[tokio::main]
async fn main() ->anyhow::Result<()> {


    pretty_env_logger::init();

    /**************************************************************/
    // Create a new Tokio runtime on a dedicated thread
    /**************************************************************/
    let dedicated_runtime = Arc::new(Mutex::new(None));

    // Create a dedicated Tokio runtime on a separate thread
    let dedicated_thread_handle = tokio::task::spawn_blocking({
        let dedicated_runtime = dedicated_runtime.clone();
        move || {
            *dedicated_runtime.lock().unwrap() = Some(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(NB_WORKERS)
                    .enable_all()
                    .build()
                    .expect("Failed to create dedicated runtime"),
            );
        }
    });

    // Get a handle of this dedicated thread
    dedicated_thread_handle.await.expect("Failed to spawn dedicated thread");

    /**************************************************************/
    // Initialization Chain
    /**************************************************************/
    let args_init=Args::new();
    // Todo: Ensure connection to a specific gguf tokenizer/model

    /**************************************************************/
    // Model Selection Chain
    /**************************************************************/
    // Todo : We need to have a trait with two methods ( init, and generate)
    // Todo : Use feature flags to select mistral_llm, llama,...

    /**************************************************************/
    // Initialization of the demo web page
    /**************************************************************/
    // retrieves root html page
    let index_text= fs::read_to_string("./site/index.html")?;
    // Route to retrieve the html page
    let routes_index=warp::get().map(move || warp::reply::html(index_text.clone()));

    /**************************************************************/
    // Initialization llm model
    /**************************************************************/

    // Retrieve llm package : Model, Device, Tokenizer
    // Todo: Pass args
    // these are features
    let llm_initialize: Box<dyn LLM>=   Box::new(MistralLlm);
    let llm_package=llm_initialize.initialize(args_init).unwrap();

    /**************************************************************/
    // Text Generation Route
    /**************************************************************/

    let routes_generation = warp::path("token_stream")
        .and(warp::post())
        .and(prompt_json_body())
        .map( move |prompt :Prompt| {

            // Create a new channel for each request
            let (tx, rx):(UnboundedSender<String>,UnboundedReceiver<String>)  = mpsc::unbounded_channel();
            let rx_stream = UnboundedReceiverStream::new(rx);

            let llm_package_clone=llm_package.clone();

            // Clone the Arc for the closure
            let dedicated_runtime_clone = dedicated_runtime.clone();

            // Spawn a Tokio task in the dedicated runtime for this specific channel
            let _ = dedicated_runtime_clone.lock().unwrap().as_ref().unwrap().spawn(async move {
                process_generation(llm_package_clone, prompt.query, tx).await;
            });

            let event_stream = rx_stream.map(  move |token| {
                Ok(Bytes::from(token))
            });

            event_stream

    })
        .then(handler_stream);

    /**************************************************************/
    // Launch Server
    /**************************************************************/

    warp::serve(routes_generation.or(routes_index)).run(([127, 0, 0, 1], 3030)).await;

    Ok(())
}

/*****************************************************************/
// route handlers
/*****************************************************************/

async fn handler_stream(
    body: impl Stream<Item = Result< Bytes, Infallible>> + Unpin + Send + Sync + 'static,
) -> Result<hyper::Response<Body>, Infallible> {
    let body= hyper::Body::wrap_stream(body);
    Ok(warp::reply::Response::new(body))
}


fn prompt_json_body() -> impl Filter<Extract = (Prompt,), Error = warp::Rejection> + Clone {
    warp::body::content_length_limit(1024 * 16)
        .and(warp::body::json())
}

/*****************************************************************/
// This will call the generate method for appropriate llm model
/*****************************************************************/
async fn process_generation(llm_package:LlmPackage,prompt:String,tx: UnboundedSender<String>) {
    let _ = generate(llm_package, prompt.as_str(),tx);
}