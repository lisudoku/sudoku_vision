use std::mem::swap;
use opencv::prelude::*;
use opencv::types::VectorOfVec4i;
use opencv::core::Mat;
use medians::Medianf64;
use opencv::imgproc::hough_lines_p;
#[allow(unused)]
use crate::sudoku_image_parser::display_lines;
#[allow(unused)]
use crate::sudoku_image_parser::show_image;

pub type Line = (i32, i32, i32, i32);

pub fn detect_lines_full(image: &Mat, original_image: &Mat) -> Result<(Vec<Line>, Vec<Line>), Box<dyn std::error::Error>> {
  let res = detect_lines(image, original_image, true);

  if let Err(e) = res {
    println!("Error detect_lines: {}", e);
    println!("Retrying with filter_by_size=false");
    detect_lines(image, original_image, false)
  } else {
    res
  }
}

pub fn detect_lines(image: &Mat, _original_image: &Mat, filter_by_size: bool) -> Result<(Vec<Line>, Vec<Line>), Box<dyn std::error::Error>> {
  let image_dimensions = image.size()?;
  let image_width = image_dimensions.width;

  // show_image(&image)?;

  // Find the lines using the Hough line transform
  let mut lines = VectorOfVec4i::new();
  hough_lines_p(
    &image, &mut lines, 1.0, std::f64::consts::PI / 180.0, 100, image_width as f64 / 2.0, 10.0
  )?;
  let lines: Vec<Line> = lines.into_iter().map(|line| (line[0], line[1], line[2], line[3])).collect();

  let mut horizontal = detect_horizontal_lines(&lines, image_width, filter_by_size)?;
  let mut vertical = detect_vertical_lines(&lines, filter_by_size)?;

  // display_lines(&_original_image, &horizontal, &vertical)?;

  if horizontal.len() > 10 || vertical.len() > 10 {
    return Err(Box::from("Too many lines"))
  }

  if horizontal.len() < 5 || vertical.len() < 5 {
    return Err(Box::from("Too few lines"))
  }

  if horizontal.len() < 10 {
    horizontal = extrapolate_horizontal_lines(horizontal);
  }
  if vertical.len() < 10 {
    vertical = extrapolate_vertical_lines(vertical);
  }

  println!("Horizontal count (final) = {}", horizontal.len());
  println!("Vertical count (final) = {}", vertical.len());

  // display_lines(&_original_image, &horizontal, &vertical)?;

  if horizontal.len() != 10 || vertical.len() != 10 {
    return Err(Box::from("Line counts not ok"))
  }

  return Ok((horizontal, vertical))
}

fn detect_horizontal_lines(lines: &Vec<Line>, image_width: i32, filter_by_size: bool) -> Result<Vec<Line>, Box<dyn std::error::Error>> {
  let mut horizontal: Vec<&Line> = lines
    .iter()
    .filter(|line| (line.0 - line.2).abs() > (line.1 - line.3).abs())
    .collect();
  println!("Horizontal count (initial) = {}", horizontal.len());

  horizontal.sort_by_key(|line| line.1);
  let mut horizontal: Vec<Line> = horizontal.iter().copied().enumerate().filter_map(|(index, &line)| {
    if index == 0 || line.1 - horizontal[index - 1].1 > 10 {
      return Some(line)
    }
    None
  }).collect();
  println!("Horizontal count (distance filter) = {}", horizontal.len());

  let line_widths: Vec<i32> = horizontal.iter().map(|line| (line.0 - line.2).abs()).collect();
  let median_width = Medianf64::median(
    line_widths.iter().map(|x| *x as f64).collect::<Vec<f64>>().as_slice()
  )?;

  let width_difference_threshold = median_width / 9.0 / 2.0;
  horizontal = horizontal.into_iter().filter_map(|line| {
    let mut x1 = line.0;
    let mut x2 = line.2;
    if x1 > x2 {
      swap(&mut x1, &mut x2);
    }

    let line_width: f64 = (x2 - x1) as f64;
    let width_diff = (line_width - median_width).abs();

    if filter_by_size && width_diff > width_difference_threshold {
      return None
    }

    // Some apps have horizontal lines from one end to another
    if filter_by_size && line_width >= image_width as f64 - 3.0 {
      return None
    }

    Some((x1, line.1, x2, line.3))
  }).collect();

  println!("Horizontal count (after size filter) = {}", horizontal.len());

  Ok(horizontal)
}

fn detect_vertical_lines(lines: &Vec<Line>, filter_by_size: bool) -> Result<Vec<Line>, Box<dyn std::error::Error>> {
  let mut vertical: Vec<&Line> = lines
    .iter()
    .filter(|line| (line.0 - line.2).abs() < (line.1 - line.3).abs())
    .collect();
  println!("Vertical count (initial) = {}", vertical.len());
  vertical.sort_by_key(|line| line.0);
  let mut vertical: Vec<Line> = vertical.iter().copied().enumerate().filter_map(|(index, &line)| {
    if index == 0 || line.0 - vertical[index - 1].0 > 10 {
      return Some(line)
    }
    None
  }).collect();
  println!("Vertical count (after distance filter) = {}", vertical.len());

  let line_heights: Vec<i32> = vertical.iter().map(|line| (line.1 - line.3).abs()).collect();

  let median_height = Medianf64::median(
    line_heights.iter().map(|x| *x as f64).collect::<Vec<f64>>().as_slice()
  )?;

  let height_difference_threshold = median_height / 9.0 / 2.0;

  vertical = vertical.into_iter().filter_map(|line| {
    let mut y1 = line.1;
    let mut y2 = line.3;
    if y1 > y2 {
      swap(&mut y1, &mut y2);
    }

    let line_height: f64 = (y2 - y1) as f64;
    let height_diff = (line_height - median_height).abs();

    if filter_by_size && height_diff > height_difference_threshold {
      return None
    }

    Some((line.0, y1, line.2, y2))
  }).collect();

  println!("Vertical count (after size filter) = {}", vertical.len());

  Ok(vertical)
}

fn extrapolate_horizontal_lines(mut horizontal: Vec<Line>) -> Vec<Line> {
  println!("Extrapolating horizontal lines");
  let min_gap = horizontal.windows(2).map(|pair| (pair[0].1 - pair[1].1).abs()).min().unwrap();
  let extra_lines: Vec<Line> = horizontal.windows(2).flat_map(|pair| {
    let line = pair[0];
    let gap = (pair[0].1 - pair[1].1).abs();
    let rows_between = (gap + 10) / min_gap;
    (0..rows_between-1).map(move |index| {
      (line.0, line.1 + (index + 1) * (gap / rows_between), line.2, line.3 + (index + 1) * (gap / rows_between))
    })
  }).collect();
  horizontal.extend(extra_lines);
  horizontal.sort_by_key(|line| line.1);
  horizontal
}

fn extrapolate_vertical_lines(mut vertical: Vec<Line>) -> Vec<Line> {
  // Try to extrapolate missing vertical lines
  println!("Extrapolating vertical lines");
  let min_gap = vertical.windows(2).map(|pair| (pair[0].0 - pair[1].0).abs()).min().unwrap();
  let extra_lines: Vec<Line> = vertical.windows(2).flat_map(|pair| {
    let line = pair[0];
    let gap = (pair[0].0 - pair[1].0).abs();
    let cols_between = (gap + 10) / min_gap;
    (0..cols_between-1).map(move |index| {
      (line.0 + (index + 1) * (gap / cols_between), line.1, line.2 + (index + 1) * (gap / cols_between), line.3)
    })
  }).collect();
  vertical.extend(extra_lines);
  vertical.sort_by_key(|line| line.0);
  vertical
}
