use crate::reddit::{fetch_relevant_mentions, init_reddit_client, process_mention};

pub async fn run_scheduler(job_interval: u64) {
  loop {
    println!(">>>>>>>>>> Running job");
    let output = std::process::Command::new("sh")
      .arg("-c")
      .arg("RUN_JOB=1 ./sudoku_vision")
      .output()
      .unwrap();
    println!("Command stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    if !output.stderr.is_empty() {
      println!("Command stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    }
    tokio::time::sleep(std::time::Duration::from_secs(job_interval)).await;
  }
}

pub async fn run_job() -> Result<(), Box<dyn std::error::Error>> {
  println!("Starting Job");

  let client = match init_reddit_client().await {
    Ok(c) => c,
    Err(_) => {
      return Err(Box::from("Can't initialize reddit client"))
    }
  };

  let mentions = fetch_relevant_mentions(&client).await;

  println!("Processing {} mentions", mentions.len());
  for mention in mentions {
    let res = process_mention(&client, &mention).await;

    if let Err(e) = res {
      println!("Error for mention {}: {}", mention.id, e);
      continue
    }

    crate::discord::notify_about_mention(&mention)?;
  }

  Ok(())
}
