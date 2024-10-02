# sudoku_vision

This is a server that exposes an API for sudoku OCR.

It uses [lisudoku_ocr](https://github.com/lisudoku/lisudoku_ocr) for digit detection.

See it in action on the [lisudoku solver](http://lisudoku.xyz/solver) page.

<!-- This is a [reddit bot](https://www.reddit.com/user/sudoku-solver-bot/) that gives hints for submissions on [r/sudoku](https://www.reddit.com/r/sudoku).

ENV variables
* `JOB_INTERVAL` = how often to run the checker
* `SUBREDDIT` = name of the subreddit to check (e.g. "sudoku")
* `POST_FLAIR` = only check posts with this flair (e.g. "Request Puzzle Help")
* `POST_COUNT` = how many of the latest subreddit posts to check every time
* `BOT_COMMENT_HISTORY_LIMIT` = number of bot comments to check in order to avoid duplicate comments
* `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_PASSWORD` = reddit auth
-->

## Contribute

Join the [discord server](https://discord.gg/SGV8TQVSeT).

## Deployment

This app is deployed on [fly.io](https://fly.io/).

Deployment command: `fly deploy`
