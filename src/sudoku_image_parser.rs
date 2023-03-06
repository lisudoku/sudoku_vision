use std::cmp::max;
use lisudoku_solver::types::{FixedNumber};
use opencv::prelude::*;
use opencv::core::Vector;
use opencv::{core, imgcodecs, highgui};
use opencv::core::{Mat, Rect, Size, BORDER_CONSTANT, Point};
use opencv::types::VectorOfVectorOfPoint;
use opencv::imgproc::{
  canny, cvt_color, gaussian_blur, threshold, find_contours, bounding_rect,
  THRESH_BINARY, THRESH_BINARY_INV, THRESH_OTSU, COLOR_BGR2GRAY, RETR_TREE, CHAIN_APPROX_SIMPLE,
};
use warp::hyper::body::Bytes;
use self::line_detection::{detect_lines_full, Line};
#[cfg(test)]
use lisudoku_solver::types::SudokuGrid;

mod tesseract;
mod line_detection;

pub fn parse_image_at_path(image_path: &str) -> Result<Vec<FixedNumber>, Box<dyn std::error::Error>> {
  let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
  parse_image_from_object_full(&image)
}

pub fn parse_image_from_bytes(image_data: &Bytes) -> Result<Vec<FixedNumber>, Box<dyn std::error::Error>> {
  let image_vec = Vector::<u8>::from_iter(image_data.clone());
  let image = imgcodecs::imdecode(&image_vec, imgcodecs::IMREAD_COLOR)?;

  parse_image_from_object_full(&image)
}

pub fn parse_image_from_object_full(image: &Mat) -> Result<Vec<FixedNumber>, Box<dyn std::error::Error>> {
  let res = parse_image_from_object(image, false);

  if let Err(e) = res {
    println!("Error parse_image_from_object: {}", e);
    println!("Retrying with use_bw=true");
    parse_image_from_object(image, true)
  } else {
    res
  }
}

pub fn parse_image_from_object(image: &Mat, use_bw: bool) -> Result<Vec<FixedNumber>, Box<dyn std::error::Error>> {
  // Crop image to fix phone screenshots
  let cropped_image = crop_image(image)?;

  // show_image(&cropped_image)?;

  // Convert the image to grayscale
  let mut gray_image = Mat::default();
  cvt_color(&cropped_image, &mut gray_image, COLOR_BGR2GRAY, 0)?;

  // Convert to black and white to detect edges better
  let bw_image = if use_bw {
    convert_to_bw(&gray_image).unwrap()
  } else {
    gray_image.clone()
  };

  // show_image(&bw_image)?;

  // Apply Gaussian blur to the image
  let mut blurred_image = Mat::default();
  gaussian_blur(&bw_image, &mut blurred_image, Size::new(1, 1), 0.0, 0.0, BORDER_CONSTANT)?;

  // Apply Canny edge detection to the image
  let mut edges_image = Mat::default();
  canny(&blurred_image, &mut edges_image, 50.0, 150.0, 3, false)?;

  // show_image(&edges_image)?;

  let (horizontal_lines, vertical_lines) = detect_lines_full(&edges_image, &cropped_image)?;

  let mut given_digits: Vec<FixedNumber> = vec![];
  for row in 0..9 {
    for col in 0..9 {
      let digit = process_cell(row, col, &gray_image, &horizontal_lines, &vertical_lines)?;
      if let Some(value) = digit {
        given_digits.push(FixedNumber::new(row, col, value));
      }
    }
  }

  Ok(given_digits)
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

fn process_cell(row: usize, col: usize, image: &Mat, horizontal_lines: &Vec<Line>, vertical_lines: &Vec<Line>) -> Result<Option<u32>, Box<dyn std::error::Error>> {
  let y1 = horizontal_lines[row].1;
  let y2 = horizontal_lines[row + 1].1;
  let x1 = vertical_lines[col].0;
  let x2 = vertical_lines[col + 1].0;

  let padding = ((x2 - x1) as f32 * 0.1) as i32;
  let rect = Rect::new(x1 + padding, y1 + padding, x2 - x1 - 2 * padding, y2 - y1 - 2 * padding);
  let square = Mat::roi(&image, rect)?;

  let mut bw_square = Mat::default();
  threshold(&square, &mut bw_square, 128.0, 255.0, THRESH_BINARY | THRESH_OTSU)?;

  let normalized_square = convert_to_light(bw_square).unwrap();

  let mut contours = VectorOfVectorOfPoint::new();

  find_contours(
    &normalized_square,
    &mut contours,
    RETR_TREE,
    CHAIN_APPROX_SIMPLE,
    Point::new(0, 0)
  )?;

  // save_to_file(&normalized_square, "square.png");

  let dimensions = normalized_square.size()?;
  let img_width = dimensions.width;
  let img_height = dimensions.height;
  let image_max_size = max(img_width, img_height) as f32;

  let any_large_contour = contours.into_iter().any(|contour| {
    let rect = bounding_rect(&contour).unwrap();
    let max_size = max(rect.width, rect.height) as f32;

    image_max_size * 0.5 < max_size && max_size < image_max_size * 0.95
  });

  // show_image(&normalized_square)?;

  if !any_large_contour {
    // println!("No large contour found");
    return Ok(None)
  }

  detect_digit_in_square(&normalized_square)
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

fn detect_digit_in_square(square: &Mat) -> Result<Option<u32>, Box<dyn std::error::Error>> {
  let mut tess = tesseract::Tesseract::new()?;

  save_to_file(&square, "square.png");

  tess = tess.set_image("square.png")?;

  let text = tess.get_text()?;

  // dbg!(&text);
  // show_image(&square)?;

  return match text.trim().parse::<u32>() {
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
  assert_eq!(actual_numbers.len(), expected_numbers.len());
  for index in 0..actual_numbers.len() {
    let actual = actual_numbers[index];
    let expected = expected_numbers[index];
    assert_eq!(actual, expected);
  }
}

#[test]
fn check_parse_image1() {
  let image = imgcodecs::imread("src/test_images/image1.png", imgcodecs::IMREAD_COLOR).unwrap();
  let given_digits = parse_image_from_object_full(&image).unwrap();
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
}

#[test]
fn check_parse_image2() {
  let image = imgcodecs::imread("src/test_images/image2.png", imgcodecs::IMREAD_COLOR).unwrap();
  let given_digits = parse_image_from_object_full(&image).unwrap();
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
}

#[test]
fn check_parse_image3() {
  let image = imgcodecs::imread("src/test_images/image3.jpg.webp", imgcodecs::IMREAD_COLOR).unwrap();
  let given_digits = parse_image_from_object_full(&image).unwrap();
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
}

#[test]
fn check_parse_image4() {
  let image = imgcodecs::imread("src/test_images/image4.jpg", imgcodecs::IMREAD_COLOR).unwrap();
  let given_digits = parse_image_from_object_full(&image).unwrap();
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
}

#[test]
fn check_parse_image5() {
  let image = imgcodecs::imread("src/test_images/image5.webp", imgcodecs::IMREAD_COLOR).unwrap();
  let given_digits = parse_image_from_object_full(&image).unwrap();
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
}

#[test]
fn check_parse_image6() {
  let image = imgcodecs::imread("src/test_images/image6.png", imgcodecs::IMREAD_COLOR).unwrap();
  let given_digits = parse_image_from_object_full(&image).unwrap();
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
}

#[test]
fn check_parse_image7() {
  let image = imgcodecs::imread("src/test_images/image7.jpg", imgcodecs::IMREAD_COLOR).unwrap();
  let given_digits = parse_image_from_object_full(&image).unwrap();
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
}

#[test]
fn check_parse_image8() {
  let image = imgcodecs::imread("src/test_images/image8.jpg", imgcodecs::IMREAD_COLOR).unwrap();
  let given_digits = parse_image_from_object_full(&image).unwrap();
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
}

#[test]
fn check_parse_image9() {
  let image = imgcodecs::imread("src/test_images/image9.png", imgcodecs::IMREAD_COLOR).unwrap();
  let given_digits = parse_image_from_object_full(&image).unwrap();
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
}
