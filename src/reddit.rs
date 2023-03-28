use std::thread;
use core::time::Duration;
use std::{env, collections::HashSet};
use chrono::{Utc, TimeZone};
use roux::{Reddit, Subreddit, User, response::BasicThing, submission::SubmissionData, util::FeedOption};
use warp::hyper::body::Bytes;

const USERNAME: &str = "sudoku-solver-bot";
const SUBREDDIT_DEFAULT: &str = "testingground4bots";
const POST_FLAIR_DEFAULT: &str = "Question";
const USER_AGENT: &str = "linux::roux:v2.2.5 (by /u/sudoku-solver-bot)";
const POST_COUNT_DEFAULT: u32 = 10;
const BOT_COMMENT_HISTORY_LIMIT_DEFAULT: u32 = 15;

pub async fn fetch_relevant_posts() -> Vec<BasicThing<SubmissionData>> {
  println!("Fetching bot's comment history");
  let user = User::new(USERNAME);
  let bot_comment_history_limit_str = env::var("BOT_COMMENT_HISTORY_LIMIT").unwrap_or(
    BOT_COMMENT_HISTORY_LIMIT_DEFAULT.to_string()
  );
  let bot_comment_history_limit = bot_comment_history_limit_str.parse::<u32>().unwrap();
  let params = FeedOption::new().limit(bot_comment_history_limit);
  let comments = user.comments(Some(params)).await.unwrap().data.children;
  thread::sleep(Duration::from_secs(2));
  println!("Received {} comments", comments.len());

  let replied_post_ids: HashSet<String> = comments
    .into_iter()
    .map(|comment| comment.data.link_id.unwrap())
    .collect();
  println!("Found {} posts where the bot commented", replied_post_ids.len());

  println!("Fetching subreddit posts");

  let subreddit_name = env::var("SUBREDDIT").unwrap_or(SUBREDDIT_DEFAULT.to_string());
  let subreddit = Subreddit::new(&subreddit_name);

  let post_count_str = env::var("POST_COUNT").unwrap_or(POST_COUNT_DEFAULT.to_string());
  let post_count = post_count_str.parse::<u32>().unwrap();
  let posts = subreddit.latest(post_count, None).await.unwrap().data.children;
  thread::sleep(Duration::from_secs(2));
  println!("Received {} posts", posts.len());

  let relevant_posts: Vec<BasicThing<SubmissionData>> = posts
  .into_iter()
  .filter(|post| {
    let created_at = Utc.timestamp_opt(post.data.created_utc as i64, 0).unwrap();
    let duration = Utc::now().signed_duration_since(created_at);

    // Do not consider posts posted under an hour ago
    if duration.num_hours() < 1 {
      return false
    }

    if post.data.url.is_none() {
      return false
    }

    let url = post.data.url.as_ref().unwrap();
    let post_flair = env::var("POST_FLAIR").unwrap_or(POST_FLAIR_DEFAULT.to_string());

    (
      url.starts_with("https://i.redd.it/") ||
      url.starts_with("https://i.imgur.com/") ||
      url.starts_with("https://imgur.com/")
    ) &&
      post.data.link_flair_text == Some(post_flair) &&
      !replied_post_ids.contains(&post.data.name)
  }).collect();

  println!("After filtering there are {} posts remaining", relevant_posts.len());

  relevant_posts
}

pub async fn get_post_image_data(post: &SubmissionData) -> Result<Bytes, Box<dyn std::error::Error>> {
  let image_url = post.url.as_ref().unwrap();

  let image_data = reqwest::get(image_url)
    .await?
    .bytes()
    .await?;

  Ok(image_data)
}

pub async fn post_reply(post: &SubmissionData, comment_text: String) -> Result<(), Box<dyn std::error::Error>> {
  println!("Logging in");

  let client = Reddit::new(
    USER_AGENT, &env::var("REDDIT_CLIENT_ID").unwrap(), &env::var("REDDIT_CLIENT_SECRET").unwrap()
  )
    .username(USERNAME)
    .password(&env::var("REDDIT_PASSWORD").unwrap())
    .login()
    .await
    .unwrap();
  thread::sleep(Duration::from_secs(2));

  println!("Posting comment");

  let response = client.comment(comment_text.as_str(), &post.name).await.expect("Error?");

  if response.status() != 200 {
    return Err(Box::from("Comment wasn't posted for some reason"))
  }

  dbg!(&response);

  thread::sleep(Duration::from_secs(2));

  println!("Posted comment!");

  Ok(())
}
