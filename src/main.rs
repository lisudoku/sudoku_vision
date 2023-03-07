use std::env;
use byte_unit::Byte;
use warp::Filter;
use roux::{submission::SubmissionData};
use lisudoku_solver::{solver::Solver, types::{SudokuConstraints, FixedNumber, SolutionType}};

mod sudoku_image_parser;
mod reddit;
mod steps;

use crate::sudoku_image_parser::{parse_image_from_bytes, parse_image_at_path};
use crate::reddit::{post_reply, fetch_relevant_posts, get_post_image_data};

const LISUDOKU_BASE_URL: &str = "https://lisudoku.xyz";

fn compute_comment_text(given_digits: Vec<FixedNumber>) -> Result<String, Box<dyn std::error::Error>> {
  let constraints = SudokuConstraints::new(9, given_digits);
  let import_data = constraints.to_lz_string();
  let full_solve_url = format!("{}/solver?import={}", LISUDOKU_BASE_URL, import_data);

  println!("Grid:\n{}", constraints.to_grid_string());

  let mut solver = Solver::new(constraints, None).with_hint_mode();
  let solution = solver.logical_solve();

  if solution.solution_type == SolutionType::None {
    return Err(Box::from("No solution found :("))
  }
  if !solution.steps.iter().any(|step| step.is_grid_step()) {
    return Err(Box::from("No grid step found :("))
  }

  let mut text = steps::compute_steps_text(solution.steps);
  text += &format!("Puzzle import string: `{}`  \n\n", solver.constraints.to_import_string());
  text += &format!("^Full ^solve ^[here]({}).  \n", full_solve_url);
  text += &format!("^I ^am ^a ^bot. ^Reply ^to ^this ^comment ^if ^there ^are ^any ^issues.\n");

  Ok(String::from(text))
}

async fn process_post(post: &SubmissionData) -> Result<(), Box<dyn std::error::Error>> {
  println!("=============================\nProcessing post {}", post.title);

  let image_data = get_post_image_data(post).await?;
  println!("Downloaded image ({})", Byte::from_bytes(image_data.len() as u128).get_appropriate_unit(false));

  if image_data.len() > 1_000_000 {
    return Err(Box::from("Image too large, skipping"))
  }

  println!("Parsing image");
  let given_digits = parse_image_from_bytes(&image_data)?;

  println!("Composing message");

  let comment_text = compute_comment_text(given_digits)?;

  post_reply(post, comment_text).await?;

  Ok(())
}

async fn run_job() -> Result<(), Box<dyn std::error::Error>> {
  println!("Starting Job");

  let relevant_posts = fetch_relevant_posts().await;

  println!("Processing {} posts", relevant_posts.len());
  for post in relevant_posts {
    let res = process_post(&post.data).await;
    if let Err(e) = res {
      println!("Error for post {}: {}", post.data.title, e);
    }
  }

  Ok(())
}

async fn run_image(image_path: &str) -> Result<(), Box<dyn std::error::Error>> {
  println!("Parsing image at {}", image_path);

  let given_digits = parse_image_at_path(image_path)?;
  let constraints = SudokuConstraints::new(9, given_digits);
  println!("Grid:\n{}", constraints.to_grid_string());

  Ok(())
}

#[tokio::main]
async fn main() {
  if let Ok(image_path) = env::var("IMAGE_PATH") {
    run_image(&image_path).await.unwrap();
    return
  }

  if env::var("RUN_JOB").is_ok() {
    run_job().await.unwrap();
    return
  }

  println!("Starting app in server mode");

  let job_interval_str = env::var("JOB_INTERVAL").unwrap_or(String::from("60"));
  let job_interval = job_interval_str.parse::<u64>().unwrap();
  println!("JOB_INTERVAL = {}", job_interval);

  tokio::spawn({
    async move {
      loop {
        println!(">>>>>>>>>> Running job");
        let output = std::process::Command::new("sh")
          .arg("-c")
          .arg("RUN_JOB=1 ./reddit_sudoku_solver")
          .output()
          .unwrap();
        println!("Command stdout:\n{}", String::from_utf8_lossy(&output.stdout));
        if !output.stderr.is_empty() {
          println!("Command stderr:\n{}", String::from_utf8_lossy(&output.stderr));
        }
        tokio::time::sleep(std::time::Duration::from_secs(job_interval)).await;
      }
    }
  });

  println!("Starting server");

  // Set up a server (necessary for health checks)
  let routes = warp::any().map(|| "OK");
  warp::serve(routes)
      .run(([0, 0, 0, 0, 0, 0, 0, 0], 8080))
      .await;
}
