use std::ffi::CString;
use tesseract_plumbing::TessBaseApi;
use tesseract_sys::TessPageSegMode;
use tesseract_plumbing::tesseract_sys::TessPageSegMode_PSM_RAW_LINE;
#[allow(unused)]
use tesseract_plumbing::tesseract_sys::TessPageSegMode_PSM_SINGLE_CHAR;

pub struct Tesseract(TessBaseApi);

impl Tesseract {
  pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
    let mut tess = Tesseract(TessBaseApi::create());
    // Training data (languages) must be in folder /usr/local/share/tessdata/
    tess = tess.init("digits")?;
    tess = tess.set_variable("tessedit_char_whitelist", "123456789")?;
    tess = tess.set_page_seg_mode(TessPageSegMode_PSM_RAW_LINE)?;
    // Raw mode seems to work better than single char mode
    // tess = tess.set_page_seg_mode(TessPageSegMode_PSM_SINGLE_CHAR)?;

    Ok(tess)
  }

  pub fn set_image(mut self, filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
    let pix = tesseract_plumbing::leptonica_plumbing::Pix::read(&CString::new(filename)?)?;
    self.0.set_image_2(&pix);
    self.0.set_source_resolution(72);
    Ok(self)
  }

  pub fn get_text(&mut self) -> Result<String, Box<dyn std::error::Error>> {
    Ok(self
        .0
        .get_utf8_text()?
        .as_ref()
        .to_string_lossy()
        .into_owned())
  }

  fn init(mut self, language: &str) -> Result<Self, Box<dyn std::error::Error>> {
    self.0.init_2(None, Some(CString::new(language)?).as_deref())?;
    Ok(self)
  }

  fn set_variable(mut self, name: &str, value: &str) -> Result<Self, Box<dyn std::error::Error>> {
    self.0
        .set_variable(&CString::new(name)?, &CString::new(value)?)?;
    Ok(self)
  }

  fn set_page_seg_mode(mut self, mode: TessPageSegMode) -> Result<Self, Box<dyn std::error::Error>> {
    self.0.set_page_seg_mode(mode);
    Ok(self)
  }
}
