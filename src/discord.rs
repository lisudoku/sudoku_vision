use std::env;
use discord::{Discord, model::ServerId};
use roux::submission::SubmissionData;

const DISCORD_SERVER_ID: ServerId = ServerId(1054785451371806871);
const DISCORD_CHANNEL_NAME: &str = "reddit-bot";

pub fn notify_about_comment(post: &SubmissionData) -> Result<(), Box<dyn std::error::Error>> {
	let discord = Discord::from_bot_token(&env::var("DISCORD_TOKEN").expect("Expected token"))?;
  
  let server_channels = discord.get_server_channels(DISCORD_SERVER_ID)?;
  let bot_channel = server_channels.into_iter().find(|channel| channel.name == DISCORD_CHANNEL_NAME).unwrap();

  let post_url = format!("https://www.reddit.com{}", post.permalink);
  let text = format!("I just solved the puzzle in this post {}", post_url);

  discord.send_message(bot_channel.id, &text, "", false)?;

  Ok(())
}
