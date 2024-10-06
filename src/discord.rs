use std::{env, io::Cursor};
use bytes::Bytes;
use discord::{model::ChannelId, Discord};
use lisudoku_ocr::OcrResult;
use lisudoku_solver::types::SudokuConstraints;
use roux::inbox::InboxData;

// const DISCORD_SERVER_ID: ServerId = ServerId(1054785451371806871);
const DISCORD_REDDIT_BOT_CHANNEL_ID: ChannelId = ChannelId(1082919024884723843); // #reddit-bot
const DISCORD_VISION_CHANNEL_ID: ChannelId = ChannelId(1291916832197840987); // #sudoku-vision

pub fn notify_about_mention(mention: &InboxData) -> Result<(), Box<dyn std::error::Error>> {
  println!("Sending message to discord");

  let post_url = format!("https://www.reddit.com{}", mention.context);
  let message = format!("I was mentioned in this comment {}", post_url);

  let discord = Discord::from_bot_token(&env::var("DISCORD_TOKEN").expect("Expected token"))?;
  discord.send_message(DISCORD_REDDIT_BOT_CHANNEL_ID, &message, "", false)?;

  println!("Sent message to discord");

  Ok(())
}

pub async fn notify_about_vision_request(image_data_opt: Option<Bytes>, ocr_result: &Result<OcrResult, String>) -> Result<(), discord::Error> {
  let message = match ocr_result {
    Ok(res) => format!("Found grid `{}`", SudokuConstraints::new(9, res.given_digits.clone()).to_import_string()),
    Err(e) => e.to_string(),
  };

  // Lazily reusing same discord bot for a different purpose
  let discord = Discord::from_bot_token(&env::var("DISCORD_TOKEN").expect("Expected token"))?;

  let res = if let Some(image_data) = image_data_opt {
    let image_data_buffer = Cursor::new(image_data);
    discord.send_file(DISCORD_VISION_CHANNEL_ID, &message, image_data_buffer, "image.png")
  } else {
    discord.send_message(DISCORD_VISION_CHANNEL_ID, &message, "", false)
  };
  if let Err(e) = res {
    return Err(e)
  }

  Ok(())
}
