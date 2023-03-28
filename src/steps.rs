use std::collections::HashSet;
use lisudoku_solver::types::{SolutionStep, Rule, CellPosition};
use crate::sudoku_image_parser::CellCandidates;

pub fn compute_steps_text(steps: Vec<SolutionStep>, candidates: Vec<CellCandidates>) -> String {
  let relevant_steps: Vec<SolutionStep> = filter_relevant_steps(steps, candidates);
  if relevant_steps.len() == 1 {
    return format!("There is a {}  \n\n", step_description(&relevant_steps[0]))
  }

  let mut text = String::default();
  text += "Steps to make progress  \n";
  for (index, step) in relevant_steps.iter().enumerate() {
    text += "* ";
    if index == relevant_steps.len() - 1 {
      text += "Finally, there is a ";
    }
    text += &format!("{}  \n", step_description(&step));
  }
  text += "\n";
  text
}

fn filter_relevant_steps(steps: Vec<SolutionStep>, candidates_list: Vec<CellCandidates>) -> Vec<SolutionStep> {
  let mut candidates = vec![ vec![ HashSet::new(); 9 ]; 9 ];
  for CellCandidates { cell, values } in candidates_list {
    candidates[cell.row][cell.col] = values.iter().cloned().collect();
  }
  steps.into_iter().filter(|step| !is_redundand_step(step, &candidates)).collect()
}

fn is_redundand_step(step: &SolutionStep, candidates: &Vec<Vec<HashSet<u32>>>) -> bool {
  match step.rule {
    Rule::Candidates => true,
    Rule::HiddenSingle | Rule::NakedSingle | Rule::Thermo => false,
    Rule::HiddenPairs | Rule::HiddenTriples => {
      cells_only_contain_candidates(&step.cells, &step.values, candidates)
    },
    Rule::XYWing => {
      let z_value = step.values[2];
      cells_do_not_contain_candidates(&step.affected_cells, &vec![ z_value ], candidates)
    },
    Rule::CommonPeerEliminationKropki => {
      cells_do_not_contain_set(&step.affected_cells, &step.values, candidates)
    },
    Rule::XWing | Rule::ThermoCandidates | Rule::KillerCandidates |
      Rule::Killer45 | Rule::Kropki | Rule::KropkiChainCandidates | Rule::TopBottomCandidates | 
      Rule::CommonPeerElimination | Rule::Swordfish | Rule::TurbotFish |
      Rule::NakedPairs | Rule::NakedTriples |
      Rule::LockedCandidatesPairs | Rule::LockedCandidatesTriples | Rule::EmptyRectangles => {
        cells_do_not_contain_candidates(&step.affected_cells, &step.values, candidates)
    },
  }
}

fn cells_only_contain_candidates(cells: &Vec<CellPosition>, values: &Vec<u32>, candidates: &Vec<Vec<HashSet<u32>>>) -> bool {
  cells.iter().all(|&CellPosition { row, col }| {
    !candidates[row][col].is_empty() && candidates[row][col].difference(&values.iter().cloned().collect()).cloned().collect::<HashSet<u32>>().is_empty()
  })
}

fn cells_do_not_contain_candidates(cells: &Vec<CellPosition>, values: &Vec<u32>, candidates: &Vec<Vec<HashSet<u32>>>) -> bool {
  cells.iter().all(|&CellPosition { row, col }| {
    !candidates[row][col].is_empty() && candidates[row][col].intersection(&values.iter().cloned().collect()).cloned().collect::<HashSet<u32>>().is_empty()
  })
}

fn cells_do_not_contain_set(cells: &Vec<CellPosition>, values: &Vec<u32>, candidates: &Vec<Vec<HashSet<u32>>>) -> bool {
  cells.iter().enumerate().all(|(index, &CellPosition { row, col })| {
    !candidates[row][col].is_empty() && !candidates[row][col].contains(&values[index])
  })
}

fn step_description(step: &SolutionStep) -> String {
  String::from(
    format!(
      "[{}]({}) {}",
      rule_display(step.rule), rule_url(step.rule), step_details(step)
    )
  )
}

fn step_details(step: &SolutionStep) -> String {
  let cell_displays: Vec<String> = step.cells.iter().map(|cell| cell.to_string()).collect();
  let cells = cell_displays.join(", ");
  let affected_cells = step.affected_cells.iter().map(|cell| cell.to_string()).collect::<Vec<String>>().join(", ");
  let area_displays: Vec<String> = step.areas.iter().map(|area| area.to_string()).collect();
  let mut sorted_values = step.values.to_vec();
  sorted_values.sort();
  let values = sorted_values.iter().map(|val| val.to_string()).collect::<Vec<String>>().join(", ");

  match step.rule {
    Rule::HiddenSingle => {
      format!(">!{}!< in >!{}!< on cell >!{}!<", step.values[0], step.areas[0].to_string(), step.cells[0].to_string())
    },
    Rule::NakedSingle => {
      format!(">!{}!< on cell >!{}!<", step.values[0], step.cells[0].to_string())
    },
    Rule::HiddenPairs | Rule::HiddenTriples => {
      format!("of >!{}!< in >!{}!<", values, step.areas[0].to_string())
    },
    Rule::XWing => {
      format!(
        " on cells >!{} ({} and {})!< removes >!{} from {} ({} and {})!<",
        cells, area_displays[0], area_displays[1], values, affected_cells,
        area_displays[2], area_displays[3]
      )
    },
    Rule::XYWing => {
      let z_value = step.values[2];
      format!(
        "of >!{}!< with pivot at >!{}!< and pincers at >!{} and {}!< which removes >!{}!< from >!{}!<",
        values, cell_displays[0], cell_displays[1], cell_displays[2], z_value, affected_cells
      )
    },
    Rule::CommonPeerElimination => {
      format!(
        " to remove value >!{}!< from >!{}<! because it would eliminate it as candidate from >!{} (cells {})!<",
        values, affected_cells, step.areas[0].to_string(), cells
      )
    },
    Rule::TurbotFish => {
      format!(
        " on strong links >!{}-{}!< and >!{}-{}!<. Because >!{} and {}!< see each other, at least one \
        of >!{} and {}!< will be >!{}!<, so remove >!{} from cells {}!<",
        cell_displays[0], cell_displays[1], cell_displays[2], cell_displays[3],
        cell_displays[0], cell_displays[2], cell_displays[1], cell_displays[3],
        values, values, affected_cells
      )
    },
    Rule::NakedPairs | Rule::NakedTriples => {
      format!(
        "of >!{}!< in >!{}!< removes >!{} from {}!<",
        values, step.areas[0].to_string(), values, affected_cells,
      )
    },
    Rule::LockedCandidatesPairs | Rule::LockedCandidatesTriples => {
      format!(
        ">!({})!< >!{} in {}!< removes >!{}!< from >!{}!<",
        values, step.areas[0].to_string(), step.areas[1].to_string(), values, affected_cells,
      )
    },
    Rule::EmptyRectangles => {
      format!(
        "in >!{}!< that sees strong link >!{}-{}!< to remove >!{}!< from >!{}!<",
        step.areas[0].to_string(), cell_displays[0], cell_displays[1], values, affected_cells
      )
    },
    Rule::Candidates | Rule::Thermo | Rule::ThermoCandidates | Rule::KillerCandidates |
      Rule::Killer45 | Rule::Kropki | Rule::KropkiChainCandidates | Rule::TopBottomCandidates | 
      Rule::CommonPeerEliminationKropki | Rule::Swordfish => unimplemented!(),
  }
}

fn rule_display(rule: Rule) -> String {
  let s = match rule {
    Rule::NakedSingle => "Naked Single",
    Rule::HiddenSingle => "Hidden Single",
    Rule::LockedCandidatesPairs => "Box-Line Reduction",
    Rule::NakedPairs => "Naked Pair",
    Rule::HiddenPairs => "Hidden Pair",
    Rule::CommonPeerElimination => "Common Peer Elimination",
    Rule::LockedCandidatesTriples => "Box-Line Reduction",
    Rule::NakedTriples => "Naked Triple",
    Rule::HiddenTriples => "Hidden Triple",
    Rule::XWing => "X-Wing",
    Rule::XYWing => "XY-Wing",
    Rule::Swordfish => "Swordfish",
    Rule::TurbotFish => "Turbot Fish",
    Rule::EmptyRectangles => "Empty Rectangle",
    Rule::Candidates | Rule::Thermo | Rule::ThermoCandidates | Rule::KillerCandidates | Rule::Killer45 | Rule::Kropki |
      Rule::KropkiChainCandidates | Rule::TopBottomCandidates | 
      Rule::CommonPeerEliminationKropki => unimplemented!(),
  };
  String::from(s)
}

fn rule_url(rule: Rule) -> String {
  let url = match rule {
    Rule::NakedSingle => "https://hodoku.sourceforge.net/en/tech_singles.php#n1",
    Rule::HiddenSingle => "https://hodoku.sourceforge.net/en/tech_singles.php#h1",
    Rule::LockedCandidatesPairs => "https://hodoku.sourceforge.net/en/tech_intersections.php",
    Rule::NakedPairs => "https://hodoku.sourceforge.net/en/tech_naked.php#n2",
    Rule::HiddenPairs => "https://hodoku.sourceforge.net/en/tech_hidden.php#h2",
    Rule::CommonPeerElimination => "https://lisudoku.xyz/learn#CommonPeerElimination",
    Rule::LockedCandidatesTriples => "https://hodoku.sourceforge.net/en/tech_intersections.php",
    Rule::NakedTriples => "https://hodoku.sourceforge.net/en/tech_naked.php#n3",
    Rule::HiddenTriples => "https://hodoku.sourceforge.net/en/tech_hidden.php#h3",
    Rule::XWing => "https://hodoku.sourceforge.net/en/tech_fishb.php#bf2",
    Rule::XYWing => "https://hodoku.sourceforge.net/en/tech_wings.php#xy",
    Rule::Swordfish => "https://hodoku.sourceforge.net/en/tech_fishb.php#bf3",
    Rule::TurbotFish => "https://hodoku.sourceforge.net/en/tech_sdp.php#tf",
    Rule::EmptyRectangles => "https://hodoku.sourceforge.net/en/tech_sdp.php#er",
    Rule::Candidates | Rule::Thermo | Rule::ThermoCandidates | Rule::KillerCandidates | Rule::Killer45 | Rule::Kropki |
      Rule::KropkiChainCandidates | Rule::TopBottomCandidates | 
      Rule::CommonPeerEliminationKropki => unimplemented!(),
  };
  String::from(url)
}
