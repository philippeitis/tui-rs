use crate::{
    buffer::Buffer,
    layout::{Constraint, Rect},
    style::Style,
    text::Text,
    widgets::{Block, StatefulWidget, Widget},
};
use cassowary::{
    strength::{MEDIUM, REQUIRED, WEAK},
    WeightedRelation::*,
    {Expression, Solver},
};
use std::{
    collections::HashMap,
    fmt::Display,
    iter::{self, Iterator},
};
use unicode_width::UnicodeWidthStr;

#[derive(Debug, Clone)]
pub struct TableState {
    offset: usize,
    selected: Option<usize>,
}

impl Default for TableState {
    fn default() -> TableState {
        TableState {
            offset: 0,
            selected: None,
        }
    }
}

impl TableState {
    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    pub fn select(&mut self, index: Option<usize>) {
        self.selected = index;
        if index.is_none() {
            self.offset = 0;
        }
    }
}

/// Holds data to be displayed in a Table widget
#[derive(Debug, Clone)]
pub enum Row<D>
where
    D: Iterator,
    D::Item: Display,
{
    Data(D),
    StyledData(D, Style),
}

/// A widget to display data in formatted columns
///
/// # Examples
///
/// ```
/// # use tui::widgets::{Block, Borders, Table, Row};
/// # use tui::layout::Constraint;
/// # use tui::style::{Style, Color};
/// let row_style = Style::default().fg(Color::White);
/// Table::new(
///         ["Col1", "Col2", "Col3"].into_iter(),
///         vec![
///             Row::StyledData(["Row11", "Row12", "Row13"].into_iter(), row_style),
///             Row::StyledData(["Row21", "Row22", "Row23"].into_iter(), row_style),
///             Row::StyledData(["Row31", "Row32", "Row33"].into_iter(), row_style),
///             Row::Data(["Row41", "Row42", "Row43"].into_iter())
///         ].into_iter()
///     )
///     .block(Block::default().title("Table"))
///     .header_style(Style::default().fg(Color::Yellow))
///     .widths(&[Constraint::Length(5), Constraint::Length(5), Constraint::Length(10)])
///     .style(Style::default().fg(Color::White))
///     .column_spacing(1);
/// ```
#[derive(Debug, Clone)]
pub struct Table<'a, H, R> {
    /// A block to wrap the widget in
    block: Option<Block<'a>>,
    /// Base style for the widget
    style: Style,
    /// Header row for all columns
    header: H,
    /// Style for the header
    header_style: Style,
    /// Width constraints for each column
    widths: &'a [Constraint],
    /// Space between each column
    column_spacing: u16,
    /// Space between the header and the rows
    header_gap: u16,
    /// Style used to render the selected row
    highlight_style: Style,
    /// Symbol in front of the selected rom
    highlight_symbol: Option<&'a str>,
    /// Data to display in each row
    rows: R,
}

impl<'a, H, R> Default for Table<'a, H, R>
where
    H: Iterator + Default,
    R: Iterator + Default,
{
    fn default() -> Table<'a, H, R> {
        Table {
            block: None,
            style: Style::default(),
            header: H::default(),
            header_style: Style::default(),
            widths: &[],
            column_spacing: 1,
            header_gap: 1,
            highlight_style: Style::default(),
            highlight_symbol: None,
            rows: R::default(),
        }
    }
}
impl<'a, H, D, R> Table<'a, H, R>
where
    H: Iterator,
    D: Iterator,
    D::Item: Display,
    R: Iterator<Item = Row<D>>,
{
    pub fn new(header: H, rows: R) -> Table<'a, H, R> {
        Table {
            block: None,
            style: Style::default(),
            header,
            header_style: Style::default(),
            widths: &[],
            column_spacing: 1,
            header_gap: 1,
            highlight_style: Style::default(),
            highlight_symbol: None,
            rows,
        }
    }
    pub fn block(mut self, block: Block<'a>) -> Table<'a, H, R> {
        self.block = Some(block);
        self
    }

    pub fn header<II>(mut self, header: II) -> Table<'a, H, R>
    where
        II: IntoIterator<Item = H::Item, IntoIter = H>,
    {
        self.header = header.into_iter();
        self
    }

    pub fn header_style(mut self, style: Style) -> Table<'a, H, R> {
        self.header_style = style;
        self
    }

    pub fn widths(mut self, widths: &'a [Constraint]) -> Table<'a, H, R> {
        let between_0_and_100 = |&w| match w {
            Constraint::Percentage(p) => p <= 100,
            _ => true,
        };
        assert!(
            widths.iter().all(between_0_and_100),
            "Percentages should be between 0 and 100 inclusively."
        );
        self.widths = widths;
        self
    }

    pub fn rows<II>(mut self, rows: II) -> Table<'a, H, R>
    where
        II: IntoIterator<Item = Row<D>, IntoIter = R>,
    {
        self.rows = rows.into_iter();
        self
    }

    pub fn style(mut self, style: Style) -> Table<'a, H, R> {
        self.style = style;
        self
    }

    pub fn highlight_symbol(mut self, highlight_symbol: &'a str) -> Table<'a, H, R> {
        self.highlight_symbol = Some(highlight_symbol);
        self
    }

    pub fn highlight_style(mut self, highlight_style: Style) -> Table<'a, H, R> {
        self.highlight_style = highlight_style;
        self
    }

    pub fn column_spacing(mut self, spacing: u16) -> Table<'a, H, R> {
        self.column_spacing = spacing;
        self
    }

    pub fn header_gap(mut self, gap: u16) -> Table<'a, H, R> {
        self.header_gap = gap;
        self
    }
}

impl<'a, H, D, R> StatefulWidget for Table<'a, H, R>
where
    H: Iterator,
    H::Item: Display,
    D: Iterator,
    D::Item: Display,
    R: Iterator<Item = Row<D>>,
{
    type State = TableState;

    fn render(mut self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        buf.set_style(area, self.style);

        // Render block if necessary and get the drawing area
        let table_area = match self.block.take() {
            Some(b) => {
                let inner_area = b.inner(area);
                b.render(area, buf);
                inner_area
            }
            None => area,
        };

        let mut solver = Solver::new();
        let mut var_indices = HashMap::new();
        let mut ccs = Vec::new();
        let mut variables = Vec::new();
        for i in 0..self.widths.len() {
            let var = cassowary::Variable::new();
            variables.push(var);
            var_indices.insert(var, i);
        }
        for (i, constraint) in self.widths.iter().enumerate() {
            ccs.push(variables[i] | GE(WEAK) | 0.);
            ccs.push(match *constraint {
                Constraint::Length(v) => variables[i] | EQ(MEDIUM) | f64::from(v),
                Constraint::Percentage(v) => {
                    variables[i] | EQ(WEAK) | (f64::from(v * table_area.width) / 100.0)
                }
                Constraint::Ratio(n, d) => {
                    variables[i]
                        | EQ(WEAK)
                        | (f64::from(table_area.width) * f64::from(n) / f64::from(d))
                }
                Constraint::Min(v) => variables[i] | GE(WEAK) | f64::from(v),
                Constraint::Max(v) => variables[i] | LE(WEAK) | f64::from(v),
            })
        }
        solver
            .add_constraint(
                variables
                    .iter()
                    .fold(Expression::from_constant(0.), |acc, v| acc + *v)
                    | LE(REQUIRED)
                    | f64::from(
                        area.width - 2 - (self.column_spacing * (variables.len() as u16 - 1)),
                    ),
            )
            .unwrap();
        solver.add_constraints(&ccs).unwrap();
        let mut solved_widths = vec![0; variables.len()];
        for &(var, value) in solver.fetch_changes() {
            let index = var_indices[&var];
            let value = if value.is_sign_negative() {
                0
            } else {
                value.round() as u16
            };
            solved_widths[index] = value
        }

        let mut y = table_area.top();
        let mut x = table_area.left();

        // Draw header
        if y < table_area.bottom() {
            for (w, t) in solved_widths.iter().zip(self.header.by_ref()) {
                buf.set_stringn(x, y, format!("{}", t), *w as usize, self.header_style);
                x += *w + self.column_spacing;
            }
        }
        y += 1 + self.header_gap;

        // Use highlight_style only if something is selected
        let (selected, highlight_style) = match state.selected {
            Some(i) => (Some(i), self.highlight_style),
            None => (None, self.style),
        };
        let highlight_symbol = self.highlight_symbol.unwrap_or("");
        let blank_symbol = iter::repeat(" ")
            .take(highlight_symbol.width())
            .collect::<String>();

        // Draw rows
        let default_style = Style::default();
        if y < table_area.bottom() {
            let remaining = (table_area.bottom() - y) as usize;

            // Make sure the table shows the selected item
            state.offset = if let Some(selected) = selected {
                if selected >= remaining + state.offset - 1 {
                    selected + 1 - remaining
                } else if selected < state.offset {
                    selected
                } else {
                    state.offset
                }
            } else {
                0
            };
            for (i, row) in self.rows.skip(state.offset).take(remaining).enumerate() {
                let (data, style, symbol) = match row {
                    Row::Data(d) | Row::StyledData(d, _)
                        if Some(i) == state.selected.map(|s| s - state.offset) =>
                    {
                        (d, highlight_style, highlight_symbol)
                    }
                    Row::Data(d) => (d, default_style, blank_symbol.as_ref()),
                    Row::StyledData(d, s) => (d, s, blank_symbol.as_ref()),
                };
                x = table_area.left();
                for (c, (w, elt)) in solved_widths.iter().zip(data).enumerate() {
                    let s = if c == 0 {
                        format!("{}{}", symbol, elt)
                    } else {
                        format!("{}", elt)
                    };
                    buf.set_stringn(x, y + i as u16, s, *w as usize, style);
                    x += *w + self.column_spacing;
                }
            }
        }
    }
}

impl<'a, H, D, R> Widget for Table<'a, H, R>
where
    H: Iterator,
    H::Item: Display,
    D: Iterator,
    D::Item: Display,
    R: Iterator<Item = Row<D>>,
{
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut state = TableState::default();
        StatefulWidget::render(self, area, buf, &mut state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn table_invalid_percentages() {
        Table::new([""].iter(), vec![Row::Data([""].iter())].into_iter())
            .widths(&[Constraint::Percentage(110)]);
    }
}

#[derive(Default)]
pub struct Row2<'a> {
    cells: Vec<Cell<'a>>,
    height: u16,
    style: Style,
    bottom_margin: u16,
}

impl<'a> Row2<'a> {
    pub fn new<T>(cells: T) -> Self
    where
        T: IntoIterator<Item = Cell<'a>>,
    {
        Self {
            height: 0,
            cells: cells.into_iter().collect(),
            style: Style::default(),
            bottom_margin: 0,
        }
    }
    pub fn height(mut self, height: u16) -> Self {
        self.height = height;
        self
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn bottom_margin(mut self, margin: u16) -> Self {
        self.bottom_margin = margin;
        self
    }

    fn total_height(&self) -> u16 {
        self.height.saturating_add(self.bottom_margin)
    }
}

#[derive(Default)]
pub struct Cell<'a> {
    content: Text<'a>,
    style: Style,
}

impl<'a> Cell<'a> {
    pub fn new<T>(content: T) -> Self
    where
        T: Into<Text<'a>>,
    {
        Self {
            content: content.into(),
            style: Style::default(),
        }
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }
}

pub struct Table2<'a> {
    /// A block to wrap the widget in
    block: Option<Block<'a>>,
    /// Base style for the widget
    style: Style,
    /// Width constraints for each column
    widths: &'a [Constraint],
    /// Space between each column
    column_spacing: u16,
    /// Style used to render the selected row
    highlight_style: Style,
    /// Symbol in front of the selected rom
    highlight_symbol: Option<&'a str>,
    /// Optional header
    header: Option<Row2<'a>>,
    /// Data to display in each row
    rows: Vec<Row2<'a>>,
}

impl<'a> Table2<'a> {
    pub fn new<T>(rows: T) -> Self
    where
        T: IntoIterator<Item = Row2<'a>>,
    {
        Self {
            block: None,
            style: Style::default(),
            widths: &[],
            column_spacing: 1,
            highlight_style: Style::default(),
            highlight_symbol: None,
            header: None,
            rows: rows.into_iter().collect(),
        }
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    pub fn header(mut self, header: Row2<'a>) -> Self {
        self.header = Some(header);
        self
    }

    pub fn widths(mut self, widths: &'a [Constraint]) -> Self {
        let between_0_and_100 = |&w| match w {
            Constraint::Percentage(p) => p <= 100,
            _ => true,
        };
        assert!(
            widths.iter().all(between_0_and_100),
            "Percentages should be between 0 and 100 inclusively."
        );
        self.widths = widths;
        self
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn highlight_symbol(mut self, highlight_symbol: &'a str) -> Self {
        self.highlight_symbol = Some(highlight_symbol);
        self
    }

    pub fn highlight_style(mut self, highlight_style: Style) -> Self {
        self.highlight_style = highlight_style;
        self
    }

    pub fn column_spacing(mut self, spacing: u16) -> Self {
        self.column_spacing = spacing;
        self
    }

    fn get_columns_widths(&self, max_width: u16, has_selection: bool) -> Vec<u16> {
        let mut solver = Solver::new();
        let mut var_indices = HashMap::new();
        let mut ccs = Vec::new();
        let mut variables = Vec::new();
        for i in 0..self.widths.len() {
            let var = cassowary::Variable::new();
            variables.push(var);
            var_indices.insert(var, i);
        }
        let spacing_width = (variables.len() as u16).saturating_sub(1) * self.column_spacing;
        let mut available_width = max_width.saturating_sub(spacing_width);
        if has_selection {
            let highlight_symbol_width =
                self.highlight_symbol.map(|s| s.width() as u16).unwrap_or(0);
            available_width = available_width.saturating_sub(highlight_symbol_width);
        }
        for (i, constraint) in self.widths.iter().enumerate() {
            ccs.push(variables[i] | GE(WEAK) | 0.);
            ccs.push(match *constraint {
                Constraint::Length(v) => variables[i] | EQ(MEDIUM) | f64::from(v),
                Constraint::Percentage(v) => {
                    variables[i] | EQ(WEAK) | (f64::from(v * available_width) / 100.0)
                }
                Constraint::Ratio(n, d) => {
                    variables[i]
                        | EQ(WEAK)
                        | (f64::from(available_width) * f64::from(n) / f64::from(d))
                }
                Constraint::Min(v) => variables[i] | GE(WEAK) | f64::from(v),
                Constraint::Max(v) => variables[i] | LE(WEAK) | f64::from(v),
            })
        }
        solver
            .add_constraint(
                variables
                    .iter()
                    .fold(Expression::from_constant(0.), |acc, v| acc + *v)
                    | LE(REQUIRED)
                    | f64::from(available_width),
            )
            .unwrap();
        solver.add_constraints(&ccs).unwrap();
        let mut widths = vec![0; variables.len()];
        for &(var, value) in solver.fetch_changes() {
            let index = var_indices[&var];
            let value = if value.is_sign_negative() {
                0
            } else {
                value.round() as u16
            };
            widths[index] = value;
        }
        // Cassowary could still return columns widths greater than the max width when there are
        // fixed length constraints that cannot be satisfied. Therefore, we clamp the widths from
        // left to right.
        let mut available_width = max_width;
        for w in &mut widths {
            *w = available_width.min(*w);
            available_width = available_width
                .saturating_sub(*w)
                .saturating_sub(self.column_spacing);
        }
        widths
    }

    fn get_row_bounds(
        &self,
        selected: Option<usize>,
        offset: usize,
        max_height: u16,
    ) -> (usize, usize) {
        let mut start = offset;
        let mut end = offset;
        let mut height = 0;
        for item in self.rows.iter().skip(offset) {
            if height + item.height > max_height {
                break;
            }
            height += item.total_height();
            end += 1;
        }

        let selected = selected.unwrap_or(0).min(self.rows.len() - 1);
        while selected >= end {
            height = height.saturating_add(self.rows[end].total_height());
            end += 1;
            while height > max_height {
                height = height.saturating_sub(self.rows[start].total_height());
                start += 1;
            }
        }
        while selected < start {
            start -= 1;
            height = height.saturating_add(self.rows[start].total_height());
            while height > max_height {
                end -= 1;
                height = height.saturating_sub(self.rows[end].total_height());
            }
        }
        (start, end)
    }
}

impl<'a> StatefulWidget for Table2<'a> {
    type State = TableState;

    fn render(mut self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        if area.area() == 0 {
            return;
        }
        buf.set_style(area, self.style);
        if self.rows.is_empty() {
            return;
        }
        let table_area = match self.block.take() {
            Some(b) => {
                let inner_area = b.inner(area);
                b.render(area, buf);
                inner_area
            }
            None => area,
        };

        let has_selection = state.selected.is_some();
        let columns_widths = self.get_columns_widths(table_area.width, has_selection);
        let highlight_symbol = self.highlight_symbol.unwrap_or("");
        let blank_symbol = iter::repeat(" ")
            .take(highlight_symbol.width())
            .collect::<String>();
        let mut current_height = 0;
        let mut rows_height = table_area.height;

        // Draw header
        if let Some(ref header) = self.header {
            let max_header_height = table_area.height.min(header.total_height());
            buf.set_style(
                Rect {
                    x: table_area.left(),
                    y: table_area.top(),
                    width: table_area.width,
                    height: table_area.height.min(header.height),
                },
                header.style,
            );
            let mut col = table_area.left();
            if has_selection {
                col += (highlight_symbol.width() as u16).min(table_area.width);
            }
            for (width, cell) in columns_widths.iter().zip(header.cells.iter()) {
                render_cell(
                    buf,
                    cell,
                    Rect {
                        x: col,
                        y: table_area.top(),
                        width: *width,
                        height: max_header_height,
                    },
                );
                col += *width + self.column_spacing;
            }
            current_height += max_header_height;
            rows_height = rows_height.saturating_sub(max_header_height);
        }

        // Draw rows
        let (start, end) = self.get_row_bounds(state.selected, state.offset, rows_height);
        state.offset = start;
        for (i, table_row) in self
            .rows
            .iter_mut()
            .enumerate()
            .skip(state.offset)
            .take(end - start)
        {
            let (row, col) = (table_area.top() + current_height, table_area.left());
            current_height += table_row.total_height();
            let table_row_area = Rect {
                x: col,
                y: row,
                width: table_area.width,
                height: table_row.height,
            };
            buf.set_style(table_row_area, table_row.style);
            let is_selected = state.selected.map(|s| s == i).unwrap_or(false);
            let table_row_start_col = if has_selection {
                let symbol = if is_selected {
                    highlight_symbol
                } else {
                    &blank_symbol
                };
                let (col, _) =
                    buf.set_stringn(col, row, symbol, table_area.width as usize, table_row.style);
                col
            } else {
                col
            };
            let mut col = table_row_start_col;
            for (width, cell) in columns_widths.iter().zip(table_row.cells.iter()) {
                render_cell(
                    buf,
                    cell,
                    Rect {
                        x: col,
                        y: row,
                        width: *width,
                        height: table_row.height,
                    },
                );
                col += *width + self.column_spacing;
            }
            if is_selected {
                buf.set_style(table_row_area, self.highlight_style);
            }
        }
    }
}

fn render_cell(buf: &mut Buffer, cell: &Cell, area: Rect) {
    buf.set_style(area, cell.style);
    for (i, spans) in cell.content.lines.iter().enumerate() {
        if i as u16 >= area.height {
            break;
        }
        buf.set_spans(area.x, area.y + i as u16, spans, area.width);
    }
}
