use std::cmp::{min, max};
use std::collections::HashSet;
use itertools::Itertools;
use lisudoku_solver::solver::Solver;
use lisudoku_solver::solver::logical_solver::candidates::Candidates;
use lisudoku_solver::types::{FixedNumber, CellPosition, SudokuConstraints};
use lisudoku_solver::solver::logical_solver::technique::Technique;
use opencv::prelude::*;
use opencv::core::Vector;
use opencv::{core, imgcodecs, highgui};
use opencv::core::{Mat, Rect, Size, BORDER_CONSTANT, Point};
use opencv::types::VectorOfVectorOfPoint;
use opencv::imgproc::{
  canny, cvt_color, gaussian_blur, threshold, find_contours, bounding_rect,
  THRESH_BINARY, THRESH_BINARY_INV, THRESH_OTSU, COLOR_BGR2GRAY, RETR_TREE, CHAIN_APPROX_SIMPLE, COLOR_GRAY2BGR,
};
use warp::hyper::body::Bytes;
use self::line_detection::{detect_lines_full, Line};
#[cfg(test)]
use lisudoku_solver::types::SudokuGrid;

mod tesseract;
mod line_detection;

const DEBUG: bool = false;

#[derive(Debug, PartialEq)]
pub struct CellCandidates {
  pub cell: CellPosition,
  pub values: Vec<u32>,
}

impl CellCandidates {
  #[cfg(test)]
  fn new(row: usize, col: usize, values: Vec<u32>) -> CellCandidates {
    CellCandidates {
      cell: CellPosition::new(row, col),
      values,
    }
  }
}

pub fn parse_image_at_path(image_path: &str) -> Result<(Vec<FixedNumber>, Vec<CellCandidates>), Box<dyn std::error::Error>> {
  let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
  parse_image_from_object_full(&image)
}

pub fn parse_image_from_bytes(image_data: &Bytes) -> Result<(Vec<FixedNumber>, Vec<CellCandidates>), Box<dyn std::error::Error>> {
  let image_vec = Vector::<u8>::from_iter(image_data.clone());
  let image = imgcodecs::imdecode(&image_vec, imgcodecs::IMREAD_COLOR)?;

  parse_image_from_object_full(&image)
}

pub fn parse_image_from_object_full(image: &Mat) -> Result<(Vec<FixedNumber>, Vec<CellCandidates>), Box<dyn std::error::Error>> {
  let res = parse_image_from_object(image, false);

  if let Err(e) = res {
    println!("Error parse_image_from_object: {}", e);
    println!("Retrying with use_bw=true");
    parse_image_from_object(image, true)
  } else {
    res
  }
}

pub fn parse_image_from_object(image: &Mat, use_bw: bool) -> Result<(Vec<FixedNumber>, Vec<CellCandidates>), Box<dyn std::error::Error>> {
  // Crop image to fix phone screenshots
  let cropped_image = crop_image(image)?;

  // Convert the image to grayscale
  let mut gray_image = Mat::default();
  cvt_color(&cropped_image, &mut gray_image, COLOR_BGR2GRAY, 0)?;

  // Convert to black and white to detect edges better
  let bw_image = if use_bw {
    convert_to_bw(&gray_image).unwrap()
  } else {
    gray_image.clone()
  };

  // Apply Gaussian blur to the image
  let mut blurred_image = Mat::default();
  gaussian_blur(&bw_image, &mut blurred_image, Size::new(1, 1), 0.0, 0.0, BORDER_CONSTANT)?;

  // Apply Canny edge detection to the image
  let mut edges_image = Mat::default();
  canny(&blurred_image, &mut edges_image, 50.0, 150.0, 3, false)?;

  let (horizontal_lines, vertical_lines) = detect_lines_full(&edges_image, &cropped_image)?;

  let given_digits = parse_given_digits(&gray_image, &horizontal_lines, &vertical_lines)?;

  let grid_candidate_options = compute_candidate_options(&given_digits);

  let candidates = parse_candidates(&gray_image, &horizontal_lines, &vertical_lines, &grid_candidate_options)?;

  Ok((given_digits, candidates))
}

fn crop_image(image: &Mat) -> Result<Mat, Box<dyn std::error::Error>> {
  let image_dimensions = image.size()?;
  let image_width = image_dimensions.width;
  let image_height = image_dimensions.height;

  let top_padding = 100;
  let final_height = image_width * 3 / 2 - top_padding;

  if image_height <= final_height + top_padding {
    return Ok(image.clone())
  }

  // Define the region of interest (ROI) in the image
  let roi = core::Rect::new(0, top_padding, image_width, final_height);

  // Extract the ROI from the image
  let cropped_image = core::Mat::roi(&image, roi)?;

  Ok(cropped_image)
}

fn parse_given_digits(
  image: &Mat, horizontal_lines: &Vec<Line>, vertical_lines: &Vec<Line>
) -> Result<Vec<FixedNumber>, Box<dyn std::error::Error>> {
  println!("Parsing big digits");
  let mut given_digits: Vec<FixedNumber> = vec![];
  for row in 0..9 {
    for col in 0..9 {
      println!("Processing (big) {} {}", row, col);
      let ocr_result = process_cell_full(row, col, &image, &horizontal_lines, &vertical_lines, true, &HashSet::new())?;
      if let Some(value) = ocr_result.0 {
        println!("{},{} => Digit {}", row, col, value);
        given_digits.push(FixedNumber::new(row, col, value));
      }
    }
  }
  Ok(given_digits)
}

fn compute_candidate_options(given_digits: &Vec<FixedNumber>) -> Vec<Vec<HashSet<u32>>> {
  let constraints = SudokuConstraints::new(9, given_digits.clone());
  let solver = Solver::new(constraints, None);
  let steps = Candidates.run(&solver);
  let step = steps.first().unwrap();
  step.candidates.to_owned().unwrap()
}

fn parse_candidates(
  image: &Mat, horizontal_lines: &Vec<Line>, vertical_lines: &Vec<Line>, grid_candidate_options: &Vec<Vec<HashSet<u32>>>
) -> Result<Vec<CellCandidates>, Box<dyn std::error::Error>> {
  println!("Parsing candidate digits");
  let mut candidates: Vec<CellCandidates> = vec![];
  for row in 0..9 {
    for col in 0..9 {
      println!("Processing (small) {} {}", row, col);
      let cell_candidate_options = &grid_candidate_options[row][col];
      let ocr_result = process_cell_full(row, col, &image, &horizontal_lines, &vertical_lines, false, cell_candidate_options)?;
      if ocr_result.0.is_none() && !ocr_result.1.is_empty() {
        let cell_candidates = ocr_result.1;
        println!("{},{} => {:?}", row, col, cell_candidates);
        candidates.push(CellCandidates { cell: CellPosition::new(row, col), values: cell_candidates});
      }
    }
  }
  Ok(candidates)
}

fn process_cell_full(
  row: usize, col: usize, image: &Mat, horizontal_lines: &Vec<Line>, vertical_lines: &Vec<Line>,
  big_digit: bool, cell_candidate_options: &HashSet<u32>
) -> Result<(Option<u32>, Vec<u32>), Box<dyn std::error::Error>> {
  let res1 = process_cell(row, col, image, horizontal_lines, vertical_lines, big_digit, cell_candidate_options, false);

  if res1.is_ok() && res1.as_ref().unwrap().0.is_some() {
    return res1
  }

  // Try to find candidates with fixed threshold too
  let res2 = process_cell(row, col, image, horizontal_lines, vertical_lines, big_digit, cell_candidate_options, true);

  if res2.is_ok() && (res1.is_err() || res1.as_ref().unwrap().1.len() < res2.as_ref().unwrap().1.len()) {
    res2
  } else {
    res1
  }
}

fn process_cell(
  row: usize, col: usize, image: &Mat, horizontal_lines: &Vec<Line>, vertical_lines: &Vec<Line>,
  big_digit: bool, cell_candidate_options: &HashSet<u32>, use_fixed_threshold: bool
) -> Result<(Option<u32>, Vec<u32>), Box<dyn std::error::Error>> {
  let y1 = horizontal_lines[row].1;
  let y2 = horizontal_lines[row + 1].1;
  let x1 = vertical_lines[col].0;
  let x2 = vertical_lines[col + 1].0;

  let rect = Rect::new(x1, y1, x2 - x1, y2 - y1);
  let square = Mat::roi(&image, rect)?;

  let mut bw_square = Mat::default();
  if use_fixed_threshold {
    // Higher value catches more black
    // This value is sensitive... if it's too high it can make the whole square black if the square is highlighted
    // Tried different fixed thresholds, but different websites will have different background colors for
    // highlighted squares
    threshold(&square, &mut bw_square, 200.0, 255.0, THRESH_BINARY)?;
  } else {
    threshold(&square, &mut bw_square, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU)?;
  }

  let normalized_square = convert_to_light(bw_square).unwrap();

  let mut contours = VectorOfVectorOfPoint::new();

  find_contours(
    &normalized_square,
    &mut contours,
    RETR_TREE,
    CHAIN_APPROX_SIMPLE,
    Point::new(0, 0)
  )?;

  // Create an image to draw the contours on
  let mut output = opencv::core::Mat::default();
  cvt_color(&square, &mut output, COLOR_GRAY2BGR, 0)?;
  // square.copy_to(&mut output).unwrap();

  if DEBUG && !big_digit {
    show_image(&square)?;
    show_image(&normalized_square)?;
  }

  // Draw the contours on the output image
  let color = core::Scalar::new(255.0, 0.0, 0.0, 0.0); // red color
  let thickness = 1;

  opencv::imgproc::draw_contours(
    &mut output, &contours, -1, color, thickness,
    opencv::imgproc::LineTypes::LINE_8 as i32,
    &mut opencv::core::no_array(), i32::max_value(), Point::default()
  ).unwrap();

  // save_to_file(&normalized_square, "square.png");

  let dimensions = normalized_square.size()?;
  let img_width = dimensions.width;
  let img_height = dimensions.height;
  let image_max_size = max(img_width, img_height) as f32;

  let contour_rects: Vec<Rect> = find_contour_rects(contours, image_max_size);

  let (large_rects, small_rects): (Vec<_>, Vec<_>) = contour_rects.into_iter().partition(|rect| {
    let rect_size = max(rect.width, rect.height) as f32;
    rect_size > image_max_size * 0.4
  });

  if DEBUG {
    println!("Found rectangles: {} large, {} small", large_rects.len(), small_rects.len());
  }

  if large_rects.len() > 1 {
    Err(Box::from("Multiple large contours"))
  } else if large_rects.len() == 1 {
    if !big_digit {
      return Ok((None, vec![]))
    }
    // Note: Ignore small_rects in this case (for example 6 has a small circle)
    let digit = detect_digit_in_square_full(&normalized_square, big_digit, cell_candidate_options)?;
    Ok((digit, vec![]))
  } else if !small_rects.is_empty() {
    if big_digit {
      return Ok((None, vec![]))
    }

    let mut candidates: Vec<u32> = small_rects.into_iter().filter_map(|rect| {
      let pencilmark = Mat::roi(&square, rect).unwrap();
      let candidate = detect_digit_in_square_full(&pencilmark, big_digit, cell_candidate_options).unwrap_or(None);
      candidate
    }).collect();

    if candidates.iter().any(|&candidate| candidate < 1 || candidate > 9) ||
       candidates.clone().into_iter().unique().count() < candidates.len() {
      println!("Found invalid or duplicate candidates: {:?}", &candidates);
      return Ok((None, vec![]))
    }

    candidates.sort();

    Ok((None, candidates))
  } else {
    // No contours found
    Ok((None, vec![]))
  }
}

fn find_contour_rects(contours: VectorOfVectorOfPoint, image_max_size: f32) -> Vec<Rect> {
  let mut contour_rects: Vec<Rect> = contours.into_iter().filter_map(|contour| {
    let rect = bounding_rect(&contour).unwrap();
    let rect_size = max(rect.width, rect.height) as f32;
    let rect_small_side = min(rect.width, rect.height) as f32;

    if rect_small_side * 20.0 > rect_size && image_max_size * 0.1 < rect_size && rect_size < image_max_size * 0.9 {
      return Some(rect)
    }
    None
  }).collect();

  // Some digits may be split between 2 adjacent contours, try to merge them
  let mut used: Vec<bool> = vec![ false; contour_rects.len() ];
  let intersection_threshold = 2;
  contour_rects = contour_rects.iter().enumerate().filter_map(|(index, &rect)| {
    if used[index] {
      return None
    }
    let intersection = contour_rects[index+1..].iter().enumerate().find(|(_, &other_rect)| {
      let intersection_tl = Point::new(
        rect.x.max(other_rect.x),
        rect.y.max(other_rect.y),
      );
      let intersection_br = Point::new(
        (rect.x + rect.width).min(other_rect.x + other_rect.width),
        (rect.y + rect.height).min(other_rect.y + other_rect.height),
      );
      intersection_tl.x <= intersection_br.x &&
        intersection_tl.y - intersection_threshold <= intersection_br.y
    });

    if let Some((other_index, &other_rect)) = intersection {
      used[other_index + index + 1] = true;
      let intersection_tl = Point::new(
        rect.x.min(other_rect.x),
        rect.y.min(other_rect.y),
      );
      let intersection_br = Point::new(
        (rect.x + rect.width).max(other_rect.x + other_rect.width),
        (rect.y + rect.height).max(other_rect.y + other_rect.height),
      );
      let combined_rect = Rect::new(
        intersection_tl.x,
        intersection_tl.y,
        intersection_br.x - intersection_tl.x,
        intersection_br.y - intersection_tl.y,
      );
      Some(combined_rect)
    } else {
      Some(rect)
    }
  }).collect();

  contour_rects
}

fn convert_to_bw(image: &Mat) -> Result<Mat, Box<dyn std::error::Error>> {
  let is_light = is_image_light(image)?;
  let (lower_bound, thresh_type) = if is_light {
    (220.0, THRESH_BINARY)
  } else {
    (30.0, THRESH_BINARY_INV)
  };

  let mut bw_image = Mat::default();
  threshold(&image, &mut bw_image, lower_bound, 255.0, thresh_type)?;

  Ok(bw_image)
}

fn convert_to_light(square: Mat) -> Result<Mat, Box<dyn std::error::Error>> {
  if is_image_light(&square)? {
    return Ok(square)
  }

  // Invert the grayscale image
  let mut inverted_square = Mat::default();
  core::bitwise_not(&square, &mut inverted_square, &core::no_array())?;

  Ok(inverted_square)
}

fn is_image_light(image: &Mat) -> Result<bool, Box<dyn std::error::Error>> {
  // Calculate the mean intensity value of the image
  let mean_value = core::mean(&image, &core::no_array())?;

  let is_light = mean_value[0] > 128.0;

  Ok(is_light)
}

fn detect_digit_in_square_full(square: &Mat, big_digit: bool, cell_candidate_options: &HashSet<u32>) -> Result<Option<u32>, Box<dyn std::error::Error>> {
  let res = detect_digit_in_square(square, big_digit, cell_candidate_options, false);

  if let Err(e) = res {
    println!("Error detect_digit_in_square: {}", e);
    println!("Retrying with use_bw=true");
    detect_digit_in_square(square, big_digit, cell_candidate_options, true)
  } else {
    res
  }
}

fn detect_digit_in_square(square: &Mat, big_digit: bool, cell_candidate_options: &HashSet<u32>, use_bw: bool) -> Result<Option<u32>, Box<dyn std::error::Error>> {
  let bw_square = if use_bw {
    // maybe use thresh_otsu instead
    convert_to_bw(&square).unwrap()
  } else {
    square.clone()
  };

  // Even if we normalized already, some (candidate) squares could be highlighted differently
  let normalized_square = convert_to_light(bw_square).unwrap();

  let char_whitelist = &cell_candidate_options.iter().join("");
  let mut tess = tesseract::Tesseract::new(char_whitelist)?;

  save_to_file(&normalized_square, "square.png");

  tess = tess.set_image("square.png")?;

  let mut text = tess.get_text()?;

  if DEBUG && !big_digit {
    show_image(&normalized_square)?;
  }

  text = String::from(text.trim());

  if text.len() > 1 {
    return Err(Box::from("More than 1 character found in square"))
  }

  return match text.parse::<u32>() {
    Ok(val) => Ok(Some(val)),
    Err(e) => Err(Box::from(e.to_string()))
  }
}

#[allow(unused)]
fn show_image(image: &Mat) -> Result<(), Box<dyn std::error::Error>> {
  highgui::imshow("Result", &image)?;
  highgui::wait_key(0)?;
  Ok(())
}

#[allow(unused)]
fn show_images(image1: &Mat, image2: &Mat) -> Result<(), Box<dyn std::error::Error>> {
  highgui::imshow("Result1", &image1)?;
  highgui::imshow("Result2", &image2)?;
  highgui::wait_key(0)?;
  Ok(())
}

#[allow(unused)]
fn display_lines(image: &Mat, horizontal: &Vec<Line>, vertical: &Vec<Line>) -> Result<(), Box<dyn std::error::Error>> {
  // Draw the detected lines on the original image
  let mut result = Mat::from(image.clone());
  for line in horizontal {
    let pt1 = core::Point::new(line.0, line.1);
    let pt2 = core::Point::new(line.2, line.3);
    opencv::imgproc::line(&mut result, pt1, pt2, core::Scalar::new(0.0, 255.0, 0.0, 1.0), 1, opencv::imgproc::LINE_AA, 0)?;
  }
  for line in vertical {
    let pt1 = core::Point::new(line.0, line.1);
    let pt2 = core::Point::new(line.2, line.3);
    opencv::imgproc::line(&mut result, pt1, pt2, core::Scalar::new(0.0, 0.0, 255.0, 0.0), 1, opencv::imgproc::LINE_AA, 0)?;
  }
  show_image(&result)?;

  Ok(())
}

#[allow(unused)]
fn save_to_file(image: &Mat, filename: &str) {
  let params = Vector::new();
  imgcodecs::imwrite(filename, image, &params).unwrap();
}


#[cfg(test)]
fn compare_fixed_numbers(actual_numbers: Vec<FixedNumber>, expected_numbers: Vec<FixedNumber>) {
  assert_eq!(actual_numbers.len(), expected_numbers.len(), "Given digit counts don't match \n{:?}\n\n{:?}", &actual_numbers, &expected_numbers);
  for index in 0..actual_numbers.len() {
    let actual = actual_numbers[index];
    let expected = expected_numbers[index];
    assert_eq!(actual, expected, "Given digits don't match");
  }
}

#[cfg(test)]
fn compare_candidates(actual_candidates: Vec<CellCandidates>, expected_candidates: Vec<CellCandidates>) {
  assert_eq!(actual_candidates.len(), expected_candidates.len(), "Candidate counts don't match \n{:?}\n\n{:?}", &actual_candidates, &expected_candidates);
  for index in 0..actual_candidates.len() {
    let actual = &actual_candidates[index];
    let expected = &expected_candidates[index];
    assert_eq!(actual, expected, "Candidate values don't match");
  }
}

#[test]
fn check_parse_image1() {
  let image = imgcodecs::imread("src/test_images/image1.png", imgcodecs::IMREAD_COLOR).unwrap();
  let (given_digits, candidates) = parse_image_from_object_full(&image).unwrap();

  let expected_grid = vec![
    vec![ 0, 1, 6, 4, 2, 0, 9, 7, 8 ],
    vec![ 7, 0, 0, 8, 0, 0, 0, 4, 0 ],
    vec![ 9, 0, 0, 7, 0, 0, 0, 5, 0 ],
    vec![ 6, 0, 0, 5, 0, 2, 1, 8, 3 ],
    vec![ 0, 0, 2, 0, 0, 7, 4, 6, 5 ],
    vec![ 0, 0, 1, 6, 0, 0, 2, 9, 7 ],
    vec![ 0, 3, 0, 2, 5, 6, 7, 1, 9 ],
    vec![ 2, 6, 0, 0, 0, 0, 5, 3, 4 ],
    vec![ 1, 0, 0, 3, 0, 0, 8, 2, 6 ],
  ];
  let expected_given_digits = SudokuGrid::new(expected_grid).to_fixed_numbers();
  compare_fixed_numbers(given_digits, expected_given_digits);

  let expected_candidates = vec![
    CellCandidates::new(0, 0, vec![3, 5]),
    CellCandidates::new(0, 5, vec![3, 5]),
    CellCandidates::new(1, 1, vec![2, 5]),
    CellCandidates::new(1, 2, vec![3, 5]),
    CellCandidates::new(1, 4, vec![1, 3, 6, 9]),
    CellCandidates::new(1, 5, vec![1, 3, 5, 9]),
    CellCandidates::new(1, 6, vec![3, 6]),
    CellCandidates::new(1, 8, vec![1, 2]),
    CellCandidates::new(2, 1, vec![2, 4, 8]),
    CellCandidates::new(2, 2, vec![3, 4, 8]),
    CellCandidates::new(2, 4, vec![1, 3, 6]),
    CellCandidates::new(2, 5, vec![1, 3]),
    CellCandidates::new(2, 6, vec![3, 6]),
    CellCandidates::new(2, 8, vec![1, 2]),
    CellCandidates::new(3, 1, vec![4, 7, 9]),
    CellCandidates::new(3, 2, vec![4, 7, 9]),
    CellCandidates::new(3, 4, vec![4, 9]),
    CellCandidates::new(4, 0, vec![3, 8]),
    CellCandidates::new(4, 1, vec![8, 9]),
    CellCandidates::new(4, 3, vec![1, 9]),
    CellCandidates::new(4, 4, vec![1, 3, 8, 9]),
    CellCandidates::new(5, 0, vec![3, 4, 5, 8]),
    CellCandidates::new(5, 1, vec![4, 5, 8]),
    CellCandidates::new(5, 4, vec![3, 4, 8]),
    CellCandidates::new(5, 5, vec![3, 4, 8]),
    CellCandidates::new(6, 0, vec![4, 8]),
    CellCandidates::new(6, 2, vec![4, 8]),
    CellCandidates::new(7, 2, vec![7, 9]),
    CellCandidates::new(7, 3, vec![1, 9]),
    CellCandidates::new(7, 4, vec![1, 7, 8, 9]),
    CellCandidates::new(7, 5, vec![1, 8, 9]),
    CellCandidates::new(8, 1, vec![5, 7, 9]),
    CellCandidates::new(8, 2, vec![5, 7, 9]),
    CellCandidates::new(8, 4, vec![4, 7, 9]),
    CellCandidates::new(8, 5, vec![4, 9]),
  ];
  compare_candidates(candidates, expected_candidates);
}

#[test]
fn check_parse_image2() {
  let image = imgcodecs::imread("src/test_images/image2.png", imgcodecs::IMREAD_COLOR).unwrap();
  let (given_digits, candidates) = parse_image_from_object_full(&image).unwrap();

  let expected_grid = vec![
    vec![ 6, 8, 5, 2, 7, 1, 9, 3, 4 ],
    vec![ 0, 3, 0, 9, 4, 8, 0, 0, 6 ],
    vec![ 0, 0, 0, 6, 5, 3, 0, 0, 7 ],
    vec![ 5, 2, 0, 8, 1, 4, 0, 7, 0 ],
    vec![ 1, 0, 0, 5, 0, 0, 0, 0, 0 ],
    vec![ 8, 0, 0, 3, 0, 0, 4, 0, 0 ],
    vec![ 0, 0, 0, 4, 8, 2, 0, 0, 0 ],
    vec![ 0, 0, 8, 7, 3, 5, 0, 2, 0 ],
    vec![ 3, 5, 2, 1, 6, 9, 7, 4, 8 ],
  ];
  let expected_given_digits = SudokuGrid::new(expected_grid).to_fixed_numbers();
  compare_fixed_numbers(given_digits, expected_given_digits);

  let expected_candidates = vec![
    CellCandidates::new(1, 0, vec![2, 7]),
    CellCandidates::new(1, 2, vec![1, 7]),
    CellCandidates::new(1, 6, vec![1, 2, 5]),
    CellCandidates::new(1, 7, vec![1, 5]),
    CellCandidates::new(2, 0, vec![2, 4, 9]),
    CellCandidates::new(2, 1, vec![1, 4, 9]),
    CellCandidates::new(2, 2, vec![1, 4, 9]),
    CellCandidates::new(2, 6, vec![1, 2, 8]),
    CellCandidates::new(2, 7, vec![1, 8]),
    CellCandidates::new(3, 2, vec![3, 6, 9]),
    CellCandidates::new(3, 6, vec![3, 6]),
    CellCandidates::new(3, 8, vec![3, 9]),
    CellCandidates::new(4, 1, vec![4, 6, 7, 9]),
    CellCandidates::new(4, 2, vec![3, 4, 6, 7, 9]),
    CellCandidates::new(4, 4, vec![2, 9]),
    CellCandidates::new(4, 5, vec![6, 7]),
    CellCandidates::new(4, 6, vec![3, 6, 8]),
    CellCandidates::new(4, 7, vec![6, 8, 9]),
    CellCandidates::new(4, 8, vec![2, 3, 9]),
    CellCandidates::new(5, 1, vec![6, 7, 9]),
    CellCandidates::new(5, 2, vec![6, 7, 9]),
    CellCandidates::new(5, 4, vec![2, 9]),
    CellCandidates::new(5, 5, vec![6, 7]),
    CellCandidates::new(5, 7, vec![1, 5, 6, 9]),
    CellCandidates::new(5, 8, vec![1, 2, 5, 9]),
    CellCandidates::new(6, 0, vec![7, 9]),
    CellCandidates::new(6, 1, vec![1, 6, 7, 9]),
    CellCandidates::new(6, 2, vec![1, 6, 7, 9]),
    CellCandidates::new(6, 6, vec![1, 3, 5, 6]),
    CellCandidates::new(6, 7, vec![1, 5, 6, 9]),
    CellCandidates::new(6, 8, vec![1, 3, 5, 9]),
    CellCandidates::new(7, 0, vec![4, 9]),
    CellCandidates::new(7, 1, vec![1, 4, 6, 9]),
    CellCandidates::new(7, 6, vec![1, 6]),
    CellCandidates::new(7, 8, vec![1, 9]),
  ];
  compare_candidates(candidates, expected_candidates);
}

#[test]
fn check_parse_image3() {
  let image = imgcodecs::imread("src/test_images/image3.jpg.webp", imgcodecs::IMREAD_COLOR).unwrap();
  let (given_digits, candidates) = parse_image_from_object_full(&image).unwrap();

  let expected_grid = vec![
    vec![ 0, 7, 8, 5, 0, 0, 0, 0, 0 ],
    vec![ 0, 0, 3, 0, 0, 7, 8, 0, 0 ],
    vec![ 0, 0, 0, 1, 9, 8, 0, 0, 0 ],
    vec![ 0, 0, 7, 0, 0, 0, 2, 9, 1 ],
    vec![ 0, 9, 0, 0, 6, 1, 0, 4, 0 ],
    vec![ 0, 3, 1, 9, 0, 4, 0, 0, 0 ],
    vec![ 3, 0, 6, 0, 0, 2, 0, 0, 0 ],
    vec![ 0, 1, 0, 0, 0, 0, 0, 0, 4 ],
    vec![ 0, 0, 0, 0, 0, 0, 5, 0, 0 ],
  ];
  let expected_given_digits = SudokuGrid::new(expected_grid).to_fixed_numbers();
  compare_fixed_numbers(given_digits, expected_given_digits);

  let expected_candidates = vec![
    CellCandidates::new(0, 0, vec![1, 2, 4, 6, 9]),
    CellCandidates::new(0, 4, vec![2, 3, 4]),
    CellCandidates::new(0, 5, vec![3, 6]),
    CellCandidates::new(0, 6, vec![1, 3, 4, 6, 9]),
    CellCandidates::new(0, 7, vec![1, 2, 3, 6]),
    CellCandidates::new(0, 8, vec![2, 3, 6, 9]),
    CellCandidates::new(1, 0, vec![1, 2, 4, 5, 6, 9]),
    CellCandidates::new(1, 1, vec![2, 4, 5, 6]),
    CellCandidates::new(1, 3, vec![2, 4, 6]),
    CellCandidates::new(1, 4, vec![2, 4]),
    CellCandidates::new(1, 7, vec![1, 2, 5, 6]),
    CellCandidates::new(1, 8, vec![2, 5, 6, 9]),
    CellCandidates::new(2, 0, vec![2, 4, 5, 6]),
    CellCandidates::new(2, 1, vec![2, 4, 5, 6]),
    CellCandidates::new(2, 2, vec![2, 4, 5]),
    CellCandidates::new(2, 6, vec![3, 4, 6, 7]),
    CellCandidates::new(2, 7, vec![2, 3, 5, 6, 7]),
    CellCandidates::new(2, 8, vec![2, 3, 5, 6, 7]),
    CellCandidates::new(3, 0, vec![4, 5, 6, 8]),
    CellCandidates::new(3, 1, vec![4, 5, 6, 8]),
    CellCandidates::new(3, 3, vec![3, 8]),
    CellCandidates::new(3, 4, vec![3, 5, 8]),
    CellCandidates::new(3, 5, vec![3, 5]),
    CellCandidates::new(4, 0, vec![2, 5, 8]),
    CellCandidates::new(4, 2, vec![2, 5]),
    CellCandidates::new(4, 3, vec![2, 3, 7, 8]),
    CellCandidates::new(4, 6, vec![3, 7]),
    CellCandidates::new(4, 8, vec![3, 5, 7, 8]),
    CellCandidates::new(5, 0, vec![2, 5, 6, 8]),
    CellCandidates::new(5, 4, vec![2, 5, 7, 8]),
    CellCandidates::new(5, 6, vec![6, 7]),
    CellCandidates::new(5, 7, vec![5, 6, 7, 8]),
    CellCandidates::new(5, 8, vec![5, 6, 7, 8]),
    CellCandidates::new(6, 1, vec![4, 5, 8]),
    CellCandidates::new(6, 3, vec![4, 7, 8]),
    CellCandidates::new(6, 4, vec![1, 4, 5, 7, 8]),
    CellCandidates::new(6, 6, vec![1, 7, 9]),
    CellCandidates::new(6, 7, vec![1, 7, 8]),
    CellCandidates::new(6, 8, vec![7, 8, 9]),
    CellCandidates::new(7, 0, vec![2, 5, 7, 8, 9]),
    CellCandidates::new(7, 2, vec![2, 5, 9]),
    CellCandidates::new(7, 3, vec![3, 6, 7, 8]),
    CellCandidates::new(7, 4, vec![3, 5, 7, 8]),
    CellCandidates::new(7, 5, vec![3, 5, 6, 9]),
    CellCandidates::new(7, 6, vec![3, 6, 7, 9]),
    CellCandidates::new(7, 7, vec![2, 3, 6, 7, 8]),
    CellCandidates::new(8, 0, vec![2, 4, 7, 8, 9]),
    CellCandidates::new(8, 1, vec![2, 4, 8]),
    CellCandidates::new(8, 2, vec![2, 4, 9]),
    CellCandidates::new(8, 3, vec![3, 4, 6, 7, 8]),
    CellCandidates::new(8, 4, vec![1, 3, 4, 7, 8]),
    CellCandidates::new(8, 5, vec![3, 6, 9]),
    CellCandidates::new(8, 7, vec![1, 2, 3, 6, 7, 8]),
    CellCandidates::new(8, 8, vec![2, 3, 6, 7, 8, 9]),
  ];
  compare_candidates(candidates, expected_candidates);
}

#[test]
fn check_parse_image4() {
  let image = imgcodecs::imread("src/test_images/image4.jpg", imgcodecs::IMREAD_COLOR).unwrap();
  let (given_digits, candidates) = parse_image_from_object_full(&image).unwrap();

  let expected_grid = vec![
    vec![ 0, 2, 4, 0, 0, 0, 0, 0, 0 ],
    vec![ 0, 5, 0, 0, 0, 4, 3, 2, 0 ],
    vec![ 0, 8, 9, 6, 0, 0, 0, 0, 0 ],
    vec![ 8, 3, 0, 0, 0, 0, 6, 4, 1 ],
    vec![ 0, 4, 1, 8, 3, 0, 0, 0, 2 ],
    vec![ 9, 7, 0, 4, 0, 0, 5, 8, 3 ],
    vec![ 2, 1, 3, 7, 0, 0, 0, 0, 0 ],
    vec![ 0, 9, 0, 2, 4, 0, 1, 3, 0 ],
    vec![ 4, 6, 8, 0, 9, 0, 2, 5, 7 ],
  ];
  let expected_given_digits = SudokuGrid::new(expected_grid).to_fixed_numbers();
  compare_fixed_numbers(given_digits, expected_given_digits);

  let expected_candidates = vec![
    CellCandidates::new(0, 0, vec![1, 3, 6, 7]),
    CellCandidates::new(0, 3, vec![1, 3, 5, 9]),
    CellCandidates::new(0, 4, vec![1, 5, 7, 8]),
    CellCandidates::new(0, 5, vec![1, 5, 7, 9]),
    CellCandidates::new(0, 6, vec![7, 8, 9]),
    CellCandidates::new(0, 7, vec![1, 6, 7, 9]),
    CellCandidates::new(0, 8, vec![5, 6, 8, 9]),
    CellCandidates::new(1, 0, vec![1, 6, 7]),
    CellCandidates::new(1, 2, vec![6, 7]),
    CellCandidates::new(1, 3, vec![1, 9]),
    CellCandidates::new(1, 4, vec![1, 7, 8]),
    CellCandidates::new(1, 8, vec![6, 8, 9]),
    CellCandidates::new(2, 0, vec![1, 3, 7]),
    CellCandidates::new(2, 4, vec![1, 2, 5, 7]),
    CellCandidates::new(2, 5, vec![1, 2, 3, 5, 7]),
    CellCandidates::new(2, 6, vec![4, 7]),
    CellCandidates::new(2, 7, vec![1, 7]),
    CellCandidates::new(2, 8, vec![4, 5]),
    CellCandidates::new(3, 2, vec![2, 5]),
    CellCandidates::new(3, 3, vec![5, 9]),
    CellCandidates::new(3, 4, vec![2, 5, 7]),
    CellCandidates::new(3, 5, vec![2, 5, 7, 9]),
    CellCandidates::new(4, 0, vec![5, 6]),
    CellCandidates::new(4, 5, vec![5, 6]),
    CellCandidates::new(4, 6, vec![7, 9]),
    CellCandidates::new(4, 7, vec![7, 9]),
    CellCandidates::new(5, 2, vec![2, 6]),
    CellCandidates::new(5, 4, vec![1, 2, 6]),
    CellCandidates::new(5, 5, vec![1, 2, 6]),
    CellCandidates::new(6, 4, vec![5, 6]),
    CellCandidates::new(6, 5, vec![5, 6, 8]),
    CellCandidates::new(6, 6, vec![4, 8, 9]),
    CellCandidates::new(6, 7, vec![6, 9]),
    CellCandidates::new(6, 8, vec![4, 6, 8, 9]),
    CellCandidates::new(7, 0, vec![5, 7]),
    CellCandidates::new(7, 2, vec![5, 7]),
    CellCandidates::new(7, 5, vec![6, 8]),
    CellCandidates::new(7, 8, vec![6, 8]),
    CellCandidates::new(8, 3, vec![1, 3]),
    CellCandidates::new(8, 5, vec![1, 3]),
  ];
  compare_candidates(candidates, expected_candidates);
}

#[test]
fn check_parse_image5() {
  let image = imgcodecs::imread("src/test_images/image5.webp", imgcodecs::IMREAD_COLOR).unwrap();
  let (given_digits, candidates) = parse_image_from_object_full(&image).unwrap();
  let expected_grid = vec![
    vec![ 0, 0, 4, 3, 8, 9, 0, 7, 6 ],
    vec![ 0, 0, 0, 4, 0, 0, 0, 3, 0 ],
    vec![ 3, 0, 0, 1, 0, 5, 8, 4, 0 ],
    vec![ 0, 4, 0, 9, 0, 0, 6, 8, 1 ],
    vec![ 0, 6, 1, 0, 4, 2, 3, 9, 5 ],
    vec![ 0, 3, 0, 0, 0, 1, 4, 2, 7 ],
    vec![ 5, 0, 0, 0, 0, 4, 9, 0, 8 ],
    vec![ 0, 0, 0, 2, 9, 0, 7, 5, 4 ],
    vec![ 4, 9, 0, 0, 0, 0, 2, 0, 3 ],
  ];
  let expected_given_digits = SudokuGrid::new(expected_grid).to_fixed_numbers();
  compare_fixed_numbers(given_digits, expected_given_digits);

  let expected_candidates = vec![
    CellCandidates::new(0, 0, vec![1, 2]),
    CellCandidates::new(0, 1, vec![1, 2, 5]),
    CellCandidates::new(0, 6, vec![1, 5]),
    CellCandidates::new(1, 0, vec![1, 2, 6, 7, 8, 9]),
    CellCandidates::new(1, 1, vec![1, 2, 5, 7, 8]),
    CellCandidates::new(1, 2, vec![2, 6, 7, 8, 9]),
    CellCandidates::new(1, 4, vec![2, 6, 7]),
    CellCandidates::new(1, 5, vec![6, 7]),
    CellCandidates::new(1, 6, vec![1, 5]),
    CellCandidates::new(1, 8, vec![2, 9]),
    CellCandidates::new(2, 1, vec![2, 7]),
    CellCandidates::new(2, 2, vec![2, 6, 7, 9]),
    CellCandidates::new(2, 4, vec![2, 6, 7]),
    CellCandidates::new(2, 8, vec![2, 9]),
    CellCandidates::new(3, 0, vec![2, 7]),
    CellCandidates::new(3, 2, vec![2, 5, 7]),
    CellCandidates::new(3, 4, vec![3, 5, 7]),
    CellCandidates::new(3, 5, vec![3, 7]),
    CellCandidates::new(4, 0, vec![7, 8]),
    CellCandidates::new(4, 3, vec![7, 8]),
    CellCandidates::new(5, 0, vec![8, 9]),
    CellCandidates::new(5, 2, vec![5, 8, 9]),
    CellCandidates::new(5, 3, vec![5, 6, 8]),
    CellCandidates::new(5, 4, vec![5, 6]),
    CellCandidates::new(6, 1, vec![1, 2, 7]),
    CellCandidates::new(6, 2, vec![2, 3, 6, 7]),
    CellCandidates::new(6, 3, vec![6, 7]),
    CellCandidates::new(6, 4, vec![1, 3, 6, 7]),
    CellCandidates::new(6, 7, vec![1, 6]),
    CellCandidates::new(7, 0, vec![1, 6, 8]),
    CellCandidates::new(7, 1, vec![1, 8]),
    CellCandidates::new(7, 2, vec![3, 6, 8]),
    CellCandidates::new(7, 5, vec![3, 6, 8]),
    CellCandidates::new(8, 2, vec![6, 7, 8]),
    CellCandidates::new(8, 3, vec![5, 7, 8]),
    CellCandidates::new(8, 4, vec![1, 5, 6, 7]),
    CellCandidates::new(8, 5, vec![6, 7, 8]),
    CellCandidates::new(8, 7, vec![1, 6]),
  ];
  compare_candidates(candidates, expected_candidates);
}

#[test]
fn check_parse_image6() {
  let image = imgcodecs::imread("src/test_images/image6.png", imgcodecs::IMREAD_COLOR).unwrap();
  let (given_digits, candidates) = parse_image_from_object_full(&image).unwrap();

  let expected_grid = vec![
    vec![ 4, 6, 0, 9, 0, 1, 3, 8, 2 ],
    vec![ 9, 3, 0, 8, 4, 0, 7, 1, 6 ],
    vec![ 0, 0, 8, 3, 0, 6, 9, 5, 4 ],
    vec![ 0, 0, 4, 0, 0, 0, 8, 0, 9 ],
    vec![ 8, 5, 0, 4, 9, 0, 0, 6, 0 ],
    vec![ 3, 9, 6, 1, 8, 0, 4, 0, 5 ],
    vec![ 6, 4, 0, 7, 0, 8, 5, 0, 0 ],
    vec![ 7, 2, 3, 5, 1, 9, 6, 4, 8 ],
    vec![ 5, 8, 0, 0, 0, 4, 0, 0, 0 ],
  ];
  let expected_given_digits = SudokuGrid::new(expected_grid).to_fixed_numbers();
  compare_fixed_numbers(given_digits, expected_given_digits);

  let expected_candidates = vec![
    CellCandidates::new(0, 2, vec![5, 7]),
    CellCandidates::new(0, 4, vec![5, 7]),
    CellCandidates::new(1, 2, vec![2, 5]),
    CellCandidates::new(1, 5, vec![2, 5]),
    CellCandidates::new(2, 0, vec![1, 2]),
    CellCandidates::new(2, 1, vec![1, 7]),
    CellCandidates::new(2, 4, vec![2, 7]),
    CellCandidates::new(3, 0, vec![1, 2]),
    CellCandidates::new(3, 1, vec![1, 7]),
    CellCandidates::new(3, 3, vec![2, 6]),
    CellCandidates::new(3, 4, vec![2, 5, 6]),
    CellCandidates::new(3, 5, vec![2, 3, 5, 7]),
    CellCandidates::new(3, 7, vec![2, 3, 7]),
    CellCandidates::new(4, 2, vec![2, 7]),
    CellCandidates::new(4, 5, vec![2, 3, 7]),
    CellCandidates::new(4, 6, vec![1, 2]),
    CellCandidates::new(4, 8, vec![1, 3, 7]),
    CellCandidates::new(5, 5, vec![2, 7]),
    CellCandidates::new(5, 7, vec![2, 7]),
    CellCandidates::new(6, 2, vec![1, 9]),
    CellCandidates::new(6, 4, vec![2, 3]),
    CellCandidates::new(6, 7, vec![2, 3, 9]),
    CellCandidates::new(6, 8, vec![1, 3]),
    CellCandidates::new(8, 2, vec![1, 9]),
    CellCandidates::new(8, 3, vec![2, 6]),
    CellCandidates::new(8, 4, vec![2, 3, 6]),
    CellCandidates::new(8, 6, vec![1, 2]),
    CellCandidates::new(8, 7, vec![2, 3, 7, 9]),
    CellCandidates::new(8, 8, vec![1, 3, 7]),
  ];
  compare_candidates(candidates, expected_candidates);
}

#[test]
fn check_parse_image7() {
  let image = imgcodecs::imread("src/test_images/image7.jpg", imgcodecs::IMREAD_COLOR).unwrap();
  let (given_digits, candidates) = parse_image_from_object_full(&image).unwrap();
  let expected_grid = vec![
    vec![ 5, 1, 0, 7, 0, 0, 3, 0, 2 ],
    vec![ 9, 0, 0, 0, 0, 0, 6, 0, 0 ],
    vec![ 8, 0, 0, 0, 0, 0, 5, 0, 0 ],
    vec![ 1, 4, 9, 3, 0, 0, 8, 5, 7 ],
    vec![ 7, 3, 0, 1, 0, 0, 2, 6, 0 ],
    vec![ 2, 6, 0, 0, 0, 0, 1, 0, 3 ],
    vec![ 4, 8, 2, 5, 0, 0, 9, 1, 6 ],
    vec![ 3, 5, 1, 6, 4, 9, 7, 2, 8 ],
    vec![ 6, 9, 7, 2, 8, 1, 4, 3, 5 ],
  ];
  let expected_given_digits = SudokuGrid::new(expected_grid).to_fixed_numbers();
  compare_fixed_numbers(given_digits, expected_given_digits);

  let expected_candidates = vec![
    CellCandidates::new(0, 2, vec![4, 6]),
    CellCandidates::new(0, 4, vec![6, 9]),
    CellCandidates::new(0, 5, vec![4, 6, 8]),
    CellCandidates::new(0, 7, vec![4, 8, 9]),
    CellCandidates::new(1, 1, vec![2, 7]),
    CellCandidates::new(1, 2, vec![3, 4]),
    CellCandidates::new(1, 3, vec![4, 8]),
    CellCandidates::new(1, 4, vec![1, 2, 3, 5]),
    CellCandidates::new(1, 5, vec![2, 3, 4, 5, 8]),
    CellCandidates::new(1, 7, vec![4, 7, 8]),
    CellCandidates::new(1, 8, vec![1, 4]),
    CellCandidates::new(2, 1, vec![2, 7]),
    CellCandidates::new(2, 2, vec![3, 4, 6]),
    CellCandidates::new(2, 3, vec![4, 9]),
    CellCandidates::new(2, 4, vec![1, 2, 3, 6, 9]),
    CellCandidates::new(2, 5, vec![2, 3, 4, 6]),
    CellCandidates::new(2, 7, vec![4, 7, 9]),
    CellCandidates::new(2, 8, vec![1, 4, 9]),
    CellCandidates::new(3, 4, vec![2, 6]),
    CellCandidates::new(3, 5, vec![2, 6]),
    CellCandidates::new(4, 2, vec![5, 8]),
    CellCandidates::new(4, 4, vec![5, 9]),
    CellCandidates::new(4, 5, vec![4, 5, 8]),
    CellCandidates::new(4, 8, vec![4, 9]),
    CellCandidates::new(5, 2, vec![5, 8]),
    CellCandidates::new(5, 3, vec![4, 8, 9]),
    CellCandidates::new(5, 4, vec![5, 7, 9]),
    CellCandidates::new(5, 5, vec![4, 5, 7, 8]),
    CellCandidates::new(5, 7, vec![4, 9]),
    CellCandidates::new(6, 4, vec![3, 7]),
    CellCandidates::new(6, 5, vec![3, 7]),
  ];
  compare_candidates(candidates, expected_candidates);
}

#[test]
fn check_parse_image8() {
  let image = imgcodecs::imread("src/test_images/image8.jpg", imgcodecs::IMREAD_COLOR).unwrap();
  let (given_digits, candidates) = parse_image_from_object_full(&image).unwrap();
  let expected_grid = vec![
    vec![ 0, 0, 0, 0, 0, 5, 0, 0, 6 ],
    vec![ 5, 0, 6, 0, 0, 8, 0, 0, 9 ],
    vec![ 0, 3, 0, 0, 0, 7, 0, 5, 0 ],
    vec![ 4, 5, 2, 8, 9, 1, 7, 6, 3 ],
    vec![ 3, 9, 7, 0, 0, 2, 0, 1, 0 ],
    vec![ 6, 1, 8, 7, 5, 3, 0, 0, 4 ],
    vec![ 7, 0, 1, 0, 0, 9, 0, 0, 0 ],
    vec![ 9, 0, 3, 0, 8, 0, 0, 0, 7 ],
    vec![ 2, 0, 5, 0, 7, 0, 0, 0, 0 ],
  ];
  let expected_given_digits = SudokuGrid::new(expected_grid).to_fixed_numbers();
  compare_fixed_numbers(given_digits, expected_given_digits);

  let expected_candidates = vec![
    CellCandidates::new(0, 0, vec![1, 8]),
    CellCandidates::new(0, 1, vec![2, 7]),
    CellCandidates::new(0, 2, vec![4, 9]),
    CellCandidates::new(0, 3, vec![2, 3, 4, 9]),
    CellCandidates::new(0, 4, vec![1, 2, 3, 4]),
    CellCandidates::new(0, 6, vec![1, 2, 3, 4, 8]),
    CellCandidates::new(0, 7, vec![2, 3, 4, 7, 8]),
    CellCandidates::new(1, 1, vec![2, 7]),
    CellCandidates::new(1, 3, vec![2, 3, 4]),
    CellCandidates::new(1, 4, vec![1, 2, 3, 4]),
    CellCandidates::new(1, 6, vec![1, 2, 3, 4]),
    CellCandidates::new(1, 7, vec![2, 3, 4, 7]),
    CellCandidates::new(2, 0, vec![1, 8]),
    CellCandidates::new(2, 2, vec![4, 9]),
    CellCandidates::new(2, 3, vec![2, 4, 6, 9]),
    CellCandidates::new(2, 4, vec![1, 2, 4, 6]),
    CellCandidates::new(2, 6, vec![1, 2, 4, 8]),
    CellCandidates::new(2, 8, vec![1, 2, 8]),
    CellCandidates::new(4, 3, vec![4, 6]),
    CellCandidates::new(4, 4, vec![4, 6]),
    CellCandidates::new(4, 6, vec![5, 8]),
    CellCandidates::new(4, 8, vec![5, 8]),
    CellCandidates::new(5, 6, vec![2, 9]),
    CellCandidates::new(5, 7, vec![2, 9]),
    CellCandidates::new(6, 1, vec![4, 6, 8]),
    CellCandidates::new(6, 3, vec![2, 3, 5]),
    CellCandidates::new(6, 4, vec![2, 3]),
    CellCandidates::new(6, 6, vec![2, 3, 4, 5, 6, 8]),
    CellCandidates::new(6, 7, vec![2, 3, 4, 8]),
    CellCandidates::new(6, 8, vec![2, 5, 8]),
    CellCandidates::new(7, 1, vec![4, 6]),
    CellCandidates::new(7, 3, vec![1, 2, 5]),
    CellCandidates::new(7, 5, vec![4, 6]),
    CellCandidates::new(7, 6, vec![1, 2, 4, 5, 6]),
    CellCandidates::new(7, 7, vec![2, 4]),
    CellCandidates::new(8, 1, vec![4, 6, 8]),
    CellCandidates::new(8, 3, vec![1, 3]),
    CellCandidates::new(8, 5, vec![4, 6]),
    CellCandidates::new(8, 6, vec![1, 3, 4, 6, 8, 9]),
    CellCandidates::new(8, 7, vec![3, 4, 8, 9]),
    CellCandidates::new(8, 8, vec![1, 8]),
  ];
  compare_candidates(candidates, expected_candidates);
}

#[test]
fn check_parse_image9() {
  let image = imgcodecs::imread("src/test_images/image9.png", imgcodecs::IMREAD_COLOR).unwrap();
  let (given_digits, candidates) = parse_image_from_object_full(&image).unwrap();
  let expected_grid = vec![
    vec![ 4, 7, 3, 6, 5, 2, 0, 9, 0 ],
    vec![ 0, 5, 0, 9, 4, 3, 7, 2, 6 ],
    vec![ 9, 2, 6, 1, 7, 8, 3, 4, 5 ],
    vec![ 5, 0, 0, 7, 0, 4, 9, 1, 3 ],
    vec![ 3, 0, 0, 0, 1, 9, 0, 0, 7 ],
    vec![ 0, 9, 0, 3, 0, 0, 0, 0, 4 ],
    vec![ 0, 0, 5, 2, 3, 0, 0, 7, 9 ],
    vec![ 0, 3, 0, 0, 0, 0, 0, 0, 2 ],
    vec![ 2, 0, 0, 4, 0, 0, 0, 3, 0 ],
  ];
  let expected_given_digits = SudokuGrid::new(expected_grid).to_fixed_numbers();
  compare_fixed_numbers(given_digits, expected_given_digits);

  let expected_candidates = vec![
    CellCandidates::new(0, 6, vec![1, 8]),
    CellCandidates::new(0, 8, vec![1, 8]),
    CellCandidates::new(1, 0, vec![1, 8]),
    CellCandidates::new(1, 2, vec![1, 8]),
    CellCandidates::new(3, 1, vec![6, 8]),
    CellCandidates::new(3, 2, vec![2, 8]),
    CellCandidates::new(3, 4, vec![2, 6, 8]),
    CellCandidates::new(4, 1, vec![4, 6, 8]),
    CellCandidates::new(4, 2, vec![2, 4, 8]),
    CellCandidates::new(4, 3, vec![5, 8]),
    CellCandidates::new(4, 6, vec![2, 6]),
    CellCandidates::new(4, 7, vec![5, 6, 8]),
    CellCandidates::new(5, 0, vec![1, 7]),
    CellCandidates::new(5, 2, vec![1, 7]),
    CellCandidates::new(5, 4, vec![2, 6, 8]),
    CellCandidates::new(5, 5, vec![5, 6]),
    CellCandidates::new(5, 6, vec![2, 5, 6]),
    CellCandidates::new(5, 7, vec![5, 8]),
    CellCandidates::new(6, 0, vec![6, 8]),
    CellCandidates::new(6, 1, vec![1, 4, 8]),
    CellCandidates::new(6, 5, vec![1, 6]),
    CellCandidates::new(6, 6, vec![1, 4, 8]),
    CellCandidates::new(7, 0, vec![6, 7]),
    CellCandidates::new(7, 2, vec![4, 7, 9]),
    CellCandidates::new(7, 3, vec![5, 8]),
    CellCandidates::new(7, 4, vec![6, 8, 9]),
    CellCandidates::new(7, 5, vec![1, 5, 6, 7]),
    CellCandidates::new(7, 6, vec![1, 4]),
    CellCandidates::new(7, 7, vec![5, 6]),
    CellCandidates::new(8, 1, vec![1, 8]),
    CellCandidates::new(8, 2, vec![7, 9]),
    CellCandidates::new(8, 4, vec![6, 9]),
    CellCandidates::new(8, 5, vec![5, 6, 7]),
    CellCandidates::new(8, 6, vec![5, 6]),
    CellCandidates::new(8, 8, vec![1, 8]),
  ];
  compare_candidates(candidates, expected_candidates);
}
