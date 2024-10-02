use std::env;
use discord::{Discord, model::ServerId};
use roux::inbox::InboxData;

const DISCORD_SERVER_ID: ServerId = ServerId(1054785451371806871);
const DISCORD_CHANNEL_NAME: &str = "reddit-bot";

pub fn notify_about_mention(mention: &InboxData) -> Result<(), Box<dyn std::error::Error>> {
  println!("Sending message to discord");

  let discord = Discord::from_bot_token(&env::var("DISCORD_TOKEN").expect("Expected token"))?;
  
  let server_channels = discord.get_server_channels(DISCORD_SERVER_ID)?;
  let bot_channel = server_channels.into_iter().find(|channel| channel.name == DISCORD_CHANNEL_NAME).unwrap();

  let post_url = format!("https://www.reddit.com{}", mention.context);
  let text = format!("I was mentioned in this comment {}", post_url);

  discord.send_message(bot_channel.id, &text, "", false)?;

  println!("Sent message to discord");

  Ok(())
}
