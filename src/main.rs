use std::env;
use discord::notify_about_vision_request;
use job::run_scheduler;
use lisudoku_ocr::{get_external_image_data, parse_image_at_url, parse_image_from_bytes, OcrResult, ParseImageError};
use reqwest::Url;
use serde_json::json;
use warp::Filter;
use lisudoku_solver::types::SudokuConstraints;
use bytes::Bytes;
use warp::http::StatusCode;
use futures::StreamExt;
use warp::Buf;

mod steps;
mod reddit;
mod discord;
mod job;

#[tokio::main]
async fn main() {
  if env::var("RUN_JOB").is_ok() {
    job::run_job().await.unwrap();
    return
  }

  println!("Starting app in server mode");

  let job_interval_str = env::var("JOB_INTERVAL").unwrap_or(String::from("60"));
  let job_interval = job_interval_str.parse::<u64>().unwrap();
  println!("JOB_INTERVAL = {}", job_interval);

  tokio::spawn({
    async move {
      run_scheduler(job_interval).await
    }
  });

  println!("Starting server");

  let root_route = warp::get()
    .map(|| {
      warp::reply::json(&"Lisudoku vision is online")
    });

  let ocr_route = warp::post()
    .and(warp::path("parse_sudoku_image"))
    .and(warp::multipart::form())
    .and_then(ocr_route_handler);

  let cors = warp::cors()
    .allow_origin("http://localhost:5173")
    .allow_origin("https://lisudoku.xyz")
    .allow_methods(vec!["GET", "POST"]);

  let routes = ocr_route.or(root_route).with(cors);

  warp::serve(routes)
    .run(([0, 0, 0, 0, 0, 0, 0, 0], 8080))
    .await;
}

#[derive(Clone)]
struct ParseParams {
  file_content: Option<Bytes>,
  file_url: Option<String>,
  only_given_digits: bool,
}

fn handle_error(message: &str) -> Box<dyn warp::Reply> {
  let error_response = serde_json::json!({
    "error": message,
  });
  Box::new(warp::reply::with_status(
    warp::reply::json(&error_response),
    StatusCode::BAD_REQUEST,
  ))
}

async fn ocr_route_handler(form: warp::multipart::FormData) -> Result<Box<dyn warp::Reply>, warp::Rejection> {
  let params = match parse_params(form).await {
    Ok(p) => p,
    Err(message) => {
      return Ok(handle_error(&message))
    },
  };

  let ocr_result_result = parse_image(params.clone()).await;

  let image_data = get_image_bytes(params.clone()).await.ok();
  if let Err(e) = notify_about_vision_request(image_data, &ocr_result_result).await {
    eprintln!("Error in discord notification: {}", e);
  }

  let ocr_result = match ocr_result_result {
    Ok(res) => res,
    Err(e) => {
      return Ok(handle_error(&e.to_string()))
    }
  };

  let response = json!({
    "grid": SudokuConstraints::new(9, ocr_result.given_digits).to_import_string(),
    "candidates": ocr_result.candidates,
  });

  Ok(Box::new(warp::reply::json(&response)))
}

async fn parse_params(form: warp::multipart::FormData) -> Result<ParseParams, String> {
  let mut file_content: Option<Bytes> = None;
  let mut file_url: Option<String> = None;
  let mut only_given_digits = false;

  let mut parts = form;
  while let Some(part) = parts.next().await {
    let mut part = part.unwrap();
    if part.name() == "file_url" {
      let data = part.data().await;
      if let Some(data) = data {
        let str = String::from_utf8_lossy(data.unwrap().chunk()).to_string();
        if !str.is_empty() {
          if Url::parse(&str).is_err() {
            return Err("file_url is not a URL".to_string())
          }
          file_url = Some(str);
        }
      }
    } else if part.name() == "file_content" {
      let mut file_contents = Vec::new();
      while let Some(chunk) = part.data().await {
        match chunk {
          Ok(data) => file_contents.extend_from_slice(&data.chunk()),
          Err(_) => panic!("Something went wrong"),
        }
      }
      if !file_contents.is_empty() {
        file_content = Some(Bytes::from(file_contents));
      }
    } else if part.name() == "only_given_digits" {
      let data = part.data().await;
      if let Some(data) = data {
        only_given_digits = String::from_utf8_lossy(data.unwrap().chunk()).to_string() == "true";
      }
    }
  }

  if file_content == None && file_url == None {
    return Err("Provide file_content or file_url".to_string())
  }

  Ok(ParseParams{
    file_content,
    file_url,
    only_given_digits,
  })
}

async fn parse_image(params: ParseParams) -> Result<OcrResult, String> {
  if let Some(file_url) = params.file_url {
    return parse_image_at_url(&file_url, params.only_given_digits).await.map_err(|e| e.to_string())
  }
  if let Some(file_content) = params.file_content {
    return parse_image_from_bytes(&file_content, params.only_given_digits).map_err(|e| e.to_string())
  }
  panic!("No params available");
}

async fn get_image_bytes(params: ParseParams) -> Result<Bytes, ParseImageError> {
  if let Some(file_url) = params.file_url {
    return get_external_image_data(&file_url).await
  }
  if let Some(file_content) = params.file_content {
    return Ok(file_content)
  }
  panic!("No params available");
}
