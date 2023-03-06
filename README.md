# reddit_sudoku_solver

This is a [reddit bot](https://www.reddit.com/user/sudoku-solver-bot/) that gives hints for submissions on [r/sudoku](https://www.reddit.com/r/sudoku).

It uses OpenCV to detect lines and to split up the puzzle into squares and then Tesseract to detect digits.

ENV variables
* `JOB_INTERVAL` = how often to run the checker
* `SUBREDDIT` = name of the subreddit to check (e.g. "sudoku")
* `POST_FLAIR` = only check posts with this flair (e.g. "Request Puzzle Help")
* `POST_COUNT` = how many of the latest subreddit posts to check every time
* `BOT_COMMENT_HISTORY_LIMIT` = number of bot comments to check in order to avoid duplicate comments
* `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_PASSWORD` = reddit auth

## Contribute

Join the [discord server](https://discord.gg/SGV8TQVSeT).

## Running tests

`RUST_TEST_THREADS=1 cargo test`

## Deployment

This app is deployed on [fly.io](https://fly.io/).

Deployment command: `fly deploy`
