use std::thread;
use core::time::Duration;
use std::{env, collections::HashSet};
use chrono::{Utc, TimeZone};
use lisudoku_ocr::{parse_image_from_bytes, OcrResult};
use roux::{inbox::InboxData, response::BasicThing, submission::SubmissionData, util::FeedOption, Me, Reddit, Subreddit, User};
use warp::hyper::body::Bytes;
use byte_unit::Byte;
use lisudoku_solver::types::SudokuConstraints;

const USERNAME: &str = "sudoku-solver-bot";
const SUBREDDIT_DEFAULT: &str = "testingground4bots";
const POST_FLAIR_DEFAULT: &str = "Question";
const USER_AGENT: &str = "linux::roux:v2.2.5 (by /u/sudoku-solver-bot)";
const POST_COUNT_DEFAULT: u32 = 10;
const BOT_COMMENT_HISTORY_LIMIT_DEFAULT: u32 = 15;
const LISUDOKU_BASE_URL: &str = "https://lisudoku.xyz";

pub async fn process_post(post: &SubmissionData) -> Result<String, Box<dyn std::error::Error>> {
  println!("=============================\nProcessing post {}", post.title);

  let image_data = match get_post_image_data(post).await? {
    Some(data) => data,
    None => {
      return Ok("This is not an image post.".to_string());
    },
  };

  println!("Downloaded image ({})", Byte::from_bytes(image_data.len() as u128).get_appropriate_unit(false));

  if image_data.len() > 2_000_000 {
    return Err(Box::from("Image too large, skipping"))
  }

  println!("Parsing image");
  let ocr_result = match parse_image_from_bytes(&image_data, true) {
    Ok(res) => res,
    Err(_) => return Ok("Couldn't find a sudoku grid in this image.".to_string()),
  };

  println!("Composing message");

  let comment_text = compute_comment_text(ocr_result)?;

  println!("{}", &comment_text);

  Ok(comment_text)
}

fn compute_comment_text(ocr_result: OcrResult) -> Result<String, Box<dyn std::error::Error>> {
  let constraints = SudokuConstraints::new(9, ocr_result.given_digits);
  let import_data = constraints.to_lz_string();
  let lisudoku_play_url = format!("{}/e?import={}", LISUDOKU_BASE_URL, import_data);

  println!("Grid:\n{}", constraints.to_grid_string());
  println!("Candidates:\n{:?}", ocr_result.candidates);

  let mut text = String::new();
  text += &format!("You can solve this puzzle here: [lisudoku]({})  \n\n", lisudoku_play_url);
  text += &format!("Puzzle import string: `{}`  \n\n", constraints.to_import_string());
  text += &format!("^I ^am ^a ^bot. ^Reply ^if ^I ^made ^a ^mistake.\n");

  Ok(String::from(text))
}

pub async fn init_reddit_client() -> Result<Me, Box<dyn std::error::Error>> {
  let client = Reddit::new(
    USER_AGENT, &env::var("REDDIT_CLIENT_ID")?, &env::var("REDDIT_CLIENT_SECRET")?
  )
    .username(USERNAME)
    .password(&env::var("REDDIT_PASSWORD")?)
    .login()
    .await?;
  Ok(client)
}

pub async fn fetch_relevant_mentions(client: &Me) -> Vec<InboxData> {
  println!("Fetching bot's mentions");

  let messages = client.unread().await.unwrap().data.children;

  let mentions: Vec<InboxData> = messages.into_iter().filter_map(|message| {
    if message.data.was_comment &&
       message.data.r#type == "username_mention" {
      Some(message.data)
    } else {
      None
    }
  }).collect();

  mentions
}

pub async fn process_mention(client: &Me, mention: &InboxData) -> Result<(), Box<dyn std::error::Error>> {
  println!("Processing mention {}", mention.id);

  let tokens: Vec<_> = mention.context.split("/").collect();
  let post_id = tokens[4];
  let post_fullname = format!("t3_{}", post_id);

  println!("Fetching post {}", &post_fullname);

  let posts_response = client.get_submissions(&post_fullname).await?;
  let post = &posts_response.data.children.first().unwrap().data;

  let comment_text = process_post(post).await?;

  thread::sleep(Duration::from_secs(2));

  post_reply(client, &mention, comment_text).await?;

  println!("Marking mention as read {}", mention.name);
  client.mark_read(&mention.name).await?;

  Ok(())
}

#[allow(unused)]
pub async fn fetch_relevant_subreddit_posts() -> Vec<BasicThing<SubmissionData>> {
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

async fn get_post_image_data(post: &SubmissionData) -> Result<Option<Bytes>, Box<dyn std::error::Error>> {
  println!("Trying to fetch post image");

  if post.is_self {
    println!("Post contains no image");
    return Ok(None)
  }

  let image_url = post.url.as_ref().unwrap();

  println!("Post image url: {}", image_url);

  let image_data = reqwest::get(image_url)
    .await?
    .bytes()
    .await?;

  Ok(Some(image_data))
}

async fn post_reply(client: &Me, mention: &InboxData, comment_text: String) -> Result<(), Box<dyn std::error::Error>> {
  println!("Posting comment");

  let response = client.comment(comment_text.as_str(), &mention.name).await.expect("Error?");

  if response.status() != 200 {
    return Err(Box::from("Comment wasn't posted for some reason"))
  }

  thread::sleep(Duration::from_secs(2));

  println!("Posted comment!");

  Ok(())
}
