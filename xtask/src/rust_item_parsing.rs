//! Line-based parsing of this crate's rustfmt-formatted Rust into public items.
//! A full Rust parser would be a heavy dependency for tooling that only reads
//! formatted sources; `cargo fmt --check` guarantees the shapes handled here.

/// One public item as the reference presents it.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct PublicItem {
    pub(crate) title: String,
    pub(crate) declaration: String,
    pub(crate) documentation: String,
}

/// Item keywords that take a following name, after any `const` or `unsafe`.
const NAMED_ITEM_KEYWORDS: [&str; 7] = [
    "fn ", "struct ", "enum ", "type ", "trait ", "const ", "static ",
];

fn leading_identifier(text: &str) -> String {
    let mut identifier = String::new();
    for name_character in text.chars() {
        if name_character.is_alphanumeric() || name_character == '_' {
            identifier.push(name_character);
        } else {
            break;
        }
    }
    identifier
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Whether `identifier` appears in `source` as a whole word.
pub(crate) fn mentions_whole_word(source: &str, identifier: &str) -> bool {
    let mut search_from = 0;
    while let Some(found_at) = source[search_from..].find(identifier) {
        let identifier_start = search_from + found_at;
        let identifier_end = identifier_start + identifier.len();
        let starts_at_boundary =
            identifier_start == 0 || !is_identifier_byte(source.as_bytes()[identifier_start - 1]);
        let ends_at_boundary = identifier_end == source.len()
            || !is_identifier_byte(source.as_bytes()[identifier_end]);
        if starts_at_boundary && ends_at_boundary {
            return true;
        }
        search_from = identifier_end;
    }
    false
}

/// The text of the crate root's `pub use` statements, which decide what leaves
/// the crate: every module except `error` is private.
pub(crate) fn exported_names_text(crate_root_source: &str) -> String {
    let mut exported_text = String::new();
    let mut is_inside_statement = false;
    for source_line in crate_root_source.lines() {
        if source_line.starts_with("pub use ") {
            is_inside_statement = true;
        }
        if is_inside_statement {
            exported_text.push_str(source_line);
            exported_text.push('\n');
            is_inside_statement = !source_line.trim_end().ends_with(';');
        }
    }
    exported_text
}

/// The name of a public item starting on this line, if one does.
fn public_item_name(source_line: &str) -> Option<String> {
    let mut remainder = source_line.trim_start().strip_prefix("pub ")?;
    if let Some(after_const) = remainder.strip_prefix("const fn ") {
        return Some(leading_identifier(after_const));
    }
    if let Some(after_unsafe) = remainder.strip_prefix("unsafe ") {
        remainder = after_unsafe;
    }
    for keyword in NAMED_ITEM_KEYWORDS {
        if let Some(named_remainder) = remainder.strip_prefix(keyword) {
            return Some(leading_identifier(named_remainder));
        }
    }
    None
}

/// The type an `impl` header attaches methods to, without generics or lifetimes.
/// Trait impls return an empty name, because the trait documents their methods.
fn impl_owner(source_line: &str) -> Option<String> {
    let after_impl = source_line.strip_prefix("impl")?;
    if !after_impl.starts_with(' ') && !after_impl.starts_with('<') {
        return None;
    }
    let header = after_impl.split('{').next().unwrap_or(after_impl);
    if header.contains(" for ") {
        return Some(String::new());
    }
    let mut angle_depth = 0_i32;
    let mut previous_character = ' ';
    for (byte_position, header_character) in header.char_indices() {
        match header_character {
            '<' => angle_depth += 1,
            '>' if previous_character != '-' => angle_depth -= 1,
            _ => {}
        }
        if angle_depth == 0 && header_character.is_alphabetic() {
            return Some(leading_identifier(&header[byte_position..]));
        }
        previous_character = header_character;
    }
    Some(String::new())
}

fn doc_comment_lines_above(source_lines: &[&str], item_line_index: usize) -> Vec<String> {
    let mut doc_lines = Vec::new();
    let mut line_index = item_line_index;
    let mut is_inside_multiline_attribute = false;
    while line_index > 0 {
        line_index -= 1;
        let candidate_line = source_lines[line_index].trim_start();
        if is_inside_multiline_attribute {
            is_inside_multiline_attribute = !candidate_line.starts_with("#[");
            continue;
        }
        if candidate_line.starts_with("#[") {
            continue;
        }
        if candidate_line.ends_with(")]") {
            is_inside_multiline_attribute = true;
            continue;
        }
        let Some(doc_text) = candidate_line.strip_prefix("///") else {
            break;
        };
        doc_lines.push(doc_text.strip_prefix(' ').unwrap_or(doc_text).to_owned());
    }
    doc_lines.reverse();
    doc_lines
}

/// A function or alias declaration on one line, however rustfmt wrapped it, with
/// comments and the body removed.
fn single_line_declaration(source_lines: &[&str], item_line_index: usize) -> String {
    let mut pieces = Vec::new();
    for source_line in &source_lines[item_line_index..] {
        let code = source_line.split("//").next().unwrap_or(source_line).trim();
        if !code.is_empty() {
            pieces.push(code);
        }
        if code.contains('{') || code.ends_with(';') {
            break;
        }
    }
    let joined = pieces
        .join(" ")
        .replace("( ", "(")
        .replace(", )", ")")
        .replace(",)", ")");
    let without_body = joined.split('{').next().unwrap_or(&joined);
    let without_value = without_body.split(" = ").next().unwrap_or(without_body);
    without_value.trim_end().trim_end_matches(';').to_owned()
}

/// A struct or enum with the members a caller can use: public fields, or every
/// variant. Private fields stay hidden because they are not part of the API.
fn type_declaration(source_lines: &[&str], item_line_index: usize) -> String {
    let header_line = source_lines[item_line_index];
    let is_enum = header_line.contains(" enum ");
    if !header_line.trim_end().ends_with('{') {
        let name_end = header_line.find('(').unwrap_or(header_line.len());
        return header_line[..name_end]
            .trim()
            .trim_end_matches(';')
            .to_owned();
    }
    let member_indent = header_line.len() - header_line.trim_start().len() + 4;
    let mut members = Vec::new();
    let mut pending_docs = Vec::new();
    for source_line in &source_lines[item_line_index + 1..] {
        if source_line.trim_start().starts_with('}')
            && !source_line.starts_with(&" ".repeat(member_indent))
        {
            break;
        }
        let member_line = source_line.trim_start();
        let is_member_start = source_line.len() - member_line.len() == member_indent;
        if member_line.starts_with("///") {
            pending_docs.push(*source_line);
        } else if is_enum || (is_member_start && member_line.starts_with("pub ")) {
            members.append(&mut pending_docs);
            members.push(*source_line);
        } else {
            pending_docs.clear();
        }
    }
    let header = header_line.trim_end().trim_end_matches('{').trim_end();
    if members.is_empty() {
        return header.to_owned();
    }
    format!("{header} {{\n{}\n}}", members.join("\n"))
}

fn declaration_of(source_lines: &[&str], item_line_index: usize) -> String {
    let source_line = source_lines[item_line_index];
    if source_line.contains(" struct ") || source_line.contains(" enum ") {
        type_declaration(source_lines, item_line_index)
    } else {
        single_line_declaration(source_lines, item_line_index)
    }
}

/// Whether a line opens a test module, which ends the public part of a file.
/// A lone `#[cfg(test)]` import does not.
fn opens_test_module(source_lines: &[&str], line_index: usize) -> bool {
    source_lines[line_index].starts_with("#[cfg(test)]")
        && matches!(source_lines.get(line_index + 1), Some(following_line) if following_line.starts_with("mod "))
}

/// Every public item a source file offers callers, naming any that lack a doc
/// comment in `undocumented_items`.
pub(crate) fn public_items_in_source(
    source_text: &str,
    relative_source_path: &str,
    exported_names: &str,
    undocumented_items: &mut Vec<String>,
) -> Vec<PublicItem> {
    let source_lines: Vec<&str> = source_text.lines().collect();
    let is_public_module = relative_source_path == "error.rs";
    let mut public_items = Vec::new();
    let mut current_owner: Option<String> = None;
    for (line_index, source_line) in source_lines.iter().enumerate() {
        if opens_test_module(&source_lines, line_index) {
            break;
        }
        if let Some(owner_name) = impl_owner(source_line) {
            current_owner = Some(owner_name);
            continue;
        }
        if source_line.starts_with('}') {
            current_owner = None;
            continue;
        }
        let Some(item_name) = public_item_name(source_line) else {
            continue;
        };
        let visible_name = current_owner.as_deref().unwrap_or(&item_name);
        if visible_name.is_empty()
            || !(is_public_module || mentions_whole_word(exported_names, visible_name))
        {
            continue;
        }
        let title = match &current_owner {
            Some(owner_name) => format!("{owner_name}::{item_name}"),
            None => item_name,
        };
        let doc_lines = doc_comment_lines_above(&source_lines, line_index);
        if doc_lines.is_empty() {
            undocumented_items.push(format!("{relative_source_path}: {title}"));
        }
        public_items.push(PublicItem {
            title,
            declaration: declaration_of(&source_lines, line_index),
            documentation: markdown_from_doc_lines(&doc_lines),
        });
    }
    public_items
}

/// The module-level summary at the top of a source file.
pub(crate) fn module_summary_of(source_text: &str) -> String {
    let mut summary_lines = Vec::new();
    for source_line in source_text.lines() {
        match source_line.strip_prefix("//!") {
            Some(doc_text) => summary_lines.push(doc_text.trim_start().to_owned()),
            None => break,
        }
    }
    markdown_from_doc_lines(&summary_lines)
}

fn starts_list_entry(trimmed_line: &str) -> bool {
    trimmed_line.starts_with("- ")
        || trimmed_line.starts_with("* ")
        || matches!(trimmed_line.split_once(". "), Some((marker, _)) if !marker.is_empty() && marker.chars().all(char::is_numeric))
}

/// Rewrites rustdoc's `` [`name`] `` intra-doc links as plain code, because their
/// targets only resolve inside rustdoc.
fn without_intra_doc_links(prose_line: &str) -> String {
    let mut rewritten = prose_line.to_owned();
    while let Some(link_start) = rewritten.find("[`") {
        let Some(code_end) = rewritten[link_start..].find("`]") else {
            break;
        };
        let link_end = link_start + code_end + 2;
        if rewritten[link_end..].starts_with('(') {
            break;
        }
        let code_text = rewritten[link_start + 1..link_end - 1].to_owned();
        rewritten.replace_range(link_start..link_end, &code_text);
    }
    rewritten
}

/// Appends one prose line, joining it to the current paragraph. A trailing
/// hyphen means the source wrapped a hyphenated word, so no space goes in.
fn push_prose_line(rendered: &mut String, prose_line: &str) {
    let is_continuation = !rendered.is_empty() && !rendered.ends_with('\n');
    let ends_label = rendered.ends_with(":** ");
    if is_continuation && !ends_label && !rendered.ends_with('-') {
        rendered.push(' ');
    }
    rendered.push_str(prose_line);
}

/// Joins a doc comment's wrapped lines into Markdown paragraphs. Section headings
/// become bold labels so they cannot disturb the page's heading hierarchy.
pub(crate) fn markdown_from_doc_lines(doc_lines: &[String]) -> String {
    let mut rendered = String::new();
    let mut is_inside_code_fence = false;
    let mut is_awaiting_section_text = false;
    for doc_line in doc_lines {
        let trimmed_line = doc_line.trim();
        if trimmed_line.starts_with("```") {
            if !rendered.is_empty() && !rendered.ends_with('\n') {
                rendered.push_str("\n\n");
            }
            let is_opening_fence = !is_inside_code_fence;
            is_inside_code_fence = is_opening_fence;
            rendered.push_str(if is_opening_fence && trimmed_line == "```" {
                "```rust"
            } else {
                trimmed_line
            });
            rendered.push('\n');
        } else if is_inside_code_fence {
            if !(trimmed_line == "#" || trimmed_line.starts_with("# ")) {
                rendered.push_str(doc_line);
                rendered.push('\n');
            }
        } else if trimmed_line.is_empty() {
            if !is_awaiting_section_text {
                rendered.push_str("\n\n");
            }
        } else if trimmed_line.starts_with('#') {
            rendered.push_str(&format!(
                "\n\n**{}:** ",
                trimmed_line.trim_start_matches('#').trim()
            ));
            is_awaiting_section_text = true;
        } else {
            is_awaiting_section_text = false;
            if starts_list_entry(trimmed_line) && !rendered.is_empty() && !rendered.ends_with('\n')
            {
                rendered.push('\n');
            }
            push_prose_line(&mut rendered, &without_intra_doc_links(trimmed_line));
        }
    }
    let mut collapsed = rendered.trim().to_owned();
    while collapsed.contains("\n\n\n") {
        collapsed = collapsed.replace("\n\n\n", "\n\n");
    }
    collapsed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items_of(source_text: &str, exported_names: &str) -> (Vec<PublicItem>, Vec<String>) {
        let mut undocumented_items = Vec::new();
        let public_items = public_items_in_source(
            source_text,
            "types/sample.rs",
            exported_names,
            &mut undocumented_items,
        );
        (public_items, undocumented_items)
    }

    fn owned_lines(doc_lines: &[&str]) -> Vec<String> {
        let mut owned = Vec::new();
        for doc_line in doc_lines {
            owned.push((*doc_line).to_owned());
        }
        owned
    }

    #[test]
    fn should_title_a_method_with_its_owner_and_join_a_wrapped_signature() {
        let source = "impl Sample {\n    /// Reads it.\n    pub fn read(\n        &self,\n        axis: i32,\n    ) -> i32 {\n        axis\n    }\n}\n";

        let (public_items, undocumented_items) = items_of(source, "pub use a::Sample;");

        assert_eq!(public_items[0].title, "Sample::read");
        assert_eq!(
            public_items[0].declaration,
            "pub fn read(&self, axis: i32) -> i32"
        );
        assert_eq!(public_items[0].documentation, "Reads it.");
        assert!(undocumented_items.is_empty());
    }

    #[test]
    fn should_report_a_public_item_without_a_doc_comment() {
        let source = "impl Sample {\n    pub fn read(&self) {}\n}\n";

        let (_public_items, undocumented_items) = items_of(source, "Sample");

        assert_eq!(undocumented_items, vec!["types/sample.rs: Sample::read"]);
    }

    #[test]
    fn should_look_through_single_and_multiline_attributes_for_the_doc_comment() {
        let source = "impl Sample {\n    /// Reads it.\n    #[must_use]\n    #[allow(\n        dead_code,\n    )]\n    pub const fn read(&self) {}\n}\n";

        let (public_items, undocumented_items) = items_of(source, "Sample");

        assert_eq!(public_items[0].documentation, "Reads it.");
        assert!(undocumented_items.is_empty());
    }

    #[test]
    fn should_stop_at_an_empty_body_rather_than_swallow_the_next_lines() {
        let source = "impl Sample {\n    /// Reads it.\n    pub fn read(&self) {}\n    fn other() {\n    }\n}\n";

        let (public_items, _undocumented_items) = items_of(source, "Sample");

        assert_eq!(public_items[0].declaration, "pub fn read(&self)");
    }

    #[test]
    fn should_remove_comments_from_a_wrapped_signature() {
        let source = "/// Reads it.\npub fn read(\n    // the axis to read\n    axis: i32, // from zero\n) -> i32 {\n    axis\n}\n";

        let (public_items, _undocumented_items) = items_of(source, "read");

        assert_eq!(public_items[0].declaration, "pub fn read(axis: i32) -> i32");
    }

    #[test]
    fn should_skip_trait_impls_private_items_and_test_modules() {
        let source = "impl Drop for Sample {\n    pub fn drop(&mut self) {}\n}\nfn hidden() {}\n#[cfg(test)]\nmod tests {\n    pub fn helper() {}\n}\n";

        let (public_items, undocumented_items) = items_of(source, "Sample helper");

        assert!(public_items.is_empty());
        assert!(undocumented_items.is_empty());
    }

    #[test]
    fn should_keep_reading_past_a_conditional_import() {
        let source = "#[cfg(test)]\nuse std::fmt;\n/// Exported.\npub fn exported() {}\n";

        let (public_items, _undocumented_items) = items_of(source, "exported");

        assert_eq!(public_items.len(), 1);
    }

    #[test]
    fn should_keep_only_items_and_methods_the_crate_root_exports() {
        let source = "/// Exported.\npub fn exported() {}\n/// Internal.\npub fn internal() {}\nimpl Hidden {\n    /// Reads.\n    pub fn read(&self) {}\n}\n";

        let (public_items, undocumented_items) = items_of(source, "pub use sample::exported;");

        assert_eq!(public_items.len(), 1);
        assert_eq!(public_items[0].title, "exported");
        assert!(undocumented_items.is_empty());
    }

    #[test]
    fn should_not_treat_a_longer_exported_name_as_a_match() {
        let source = "/// Internal.\npub fn export() {}\n";

        let (public_items, _undocumented_items) = items_of(source, "pub use sample::export_all;");

        assert!(public_items.is_empty());
    }

    #[test]
    fn should_read_only_pub_use_statements_as_exports() {
        let crate_root = "//! The word hidden_name appears in prose.\nmod hidden_name;\npub use a::{\n    Listed,\n    Other,\n};\npub use b::Single;\n";

        let exported_names = exported_names_text(crate_root);

        assert!(mentions_whole_word(&exported_names, "Listed"));
        assert!(mentions_whole_word(&exported_names, "Single"));
        assert!(!mentions_whole_word(&exported_names, "hidden_name"));
    }

    #[test]
    fn should_treat_every_item_of_the_public_error_module_as_exported() {
        let mut undocumented_items = Vec::new();

        let public_items = public_items_in_source(
            "/// Sets it.\npub fn set_closure_error() {}\n",
            "error.rs",
            "",
            &mut undocumented_items,
        );

        assert_eq!(public_items.len(), 1);
    }

    #[test]
    fn should_show_enum_variants_with_their_documentation() {
        let source = "/// Kinds.\npub enum Kind {\n    /// The first.\n    First,\n    Second {\n        name: &'static str,\n    },\n}\n";

        let (public_items, _undocumented_items) = items_of(source, "Kind");

        let declaration = &public_items[0].declaration;
        assert!(declaration.starts_with("pub enum Kind {"), "{declaration}");
        assert!(declaration.contains("/// The first."), "{declaration}");
        assert!(declaration.contains("First,"), "{declaration}");
        assert!(declaration.contains("name: &'static str,"), "{declaration}");
    }

    #[test]
    fn should_show_public_fields_and_hide_private_ones() {
        let source = "/// A sample.\npub struct Sample {\n    /// Visible.\n    pub visible: i32,\n    /// Hidden.\n    hidden: i32,\n    pub(crate) internal: i32,\n}\n";

        let (public_items, _undocumented_items) = items_of(source, "Sample");

        let declaration = &public_items[0].declaration;
        assert!(declaration.contains("pub visible: i32,"), "{declaration}");
        assert!(declaration.contains("/// Visible."), "{declaration}");
        assert!(!declaration.contains("hidden"), "{declaration}");
        assert!(!declaration.contains("Hidden"), "{declaration}");
        assert!(!declaration.contains("internal"), "{declaration}");
    }

    #[test]
    fn should_show_only_the_name_of_a_struct_with_private_contents() {
        let source = "/// A handle.\npub struct Handle(raw::handle);\n/// Another.\npub struct Other {\n    secret: i32,\n}\n";

        let (public_items, _undocumented_items) = items_of(source, "Handle Other");

        assert_eq!(public_items[0].declaration, "pub struct Handle");
        assert_eq!(public_items[1].declaration, "pub struct Other");
    }

    #[test]
    fn should_recognize_public_constants_and_show_no_value() {
        let source = "/// The limit.\npub const LIMIT: usize = 4;\n";

        let (public_items, _undocumented_items) = items_of(source, "LIMIT");

        assert_eq!(public_items[0].declaration, "pub const LIMIT: usize");
    }

    #[test]
    fn should_handle_generic_and_lifetime_impl_headers() {
        assert_eq!(impl_owner("impl Sample<'_> {").as_deref(), Some("Sample"));
        assert_eq!(impl_owner("impl<T> Sample<T> {").as_deref(), Some("Sample"));
        assert_eq!(
            impl_owner("impl<F: Fn() -> i32> Sample<F> {").as_deref(),
            Some("Sample")
        );
        assert_eq!(impl_owner("impl Drop for Sample {").as_deref(), Some(""));
        assert_eq!(impl_owner("fn other() {"), None);
        assert_eq!(impl_owner("implements() {"), None);
    }

    #[test]
    fn should_join_wrapped_prose_and_turn_section_headings_into_labels() {
        let doc_lines = owned_lines(&[
            "First line",
            "second line.",
            "",
            "# Errors",
            "Fails loudly.",
        ]);

        assert_eq!(
            markdown_from_doc_lines(&doc_lines),
            "First line second line.\n\n**Errors:** Fails loudly."
        );
    }

    #[test]
    fn should_keep_a_label_with_its_text_across_a_blank_line() {
        let doc_lines = owned_lines(&["## Safety", "", "Keep it open."]);

        assert_eq!(
            markdown_from_doc_lines(&doc_lines),
            "**Safety:** Keep it open."
        );
    }

    #[test]
    fn should_keep_list_entries_on_their_own_lines() {
        let doc_lines =
            owned_lines(&["Choices:", "- first", "  continued", "- second", "1. third"]);

        assert_eq!(
            markdown_from_doc_lines(&doc_lines),
            "Choices:\n- first continued\n- second\n1. third"
        );
    }

    #[test]
    fn should_separate_a_fence_from_prose_and_label_it_and_hide_doctest_lines() {
        let doc_lines = owned_lines(&["Example:", "```", "# hidden();", "shown();", "```"]);

        assert_eq!(
            markdown_from_doc_lines(&doc_lines),
            "Example:\n\n```rust\nshown();\n```"
        );
    }

    #[test]
    fn should_rejoin_a_word_wrapped_at_its_hyphen() {
        let doc_lines = owned_lines(&["a trace-", "safe path"]);

        assert_eq!(markdown_from_doc_lines(&doc_lines), "a trace-safe path");
    }

    #[test]
    fn should_rewrite_intra_doc_links_and_keep_ordinary_links() {
        let doc_lines = owned_lines(&["See [`MlxArray::shape`] or [`docs`](https://example.com)."]);

        assert_eq!(
            markdown_from_doc_lines(&doc_lines),
            "See `MlxArray::shape` or [`docs`](https://example.com)."
        );
    }

    #[test]
    fn should_leave_no_doc_lines_as_empty_markdown() {
        assert_eq!(markdown_from_doc_lines(&[]), "");
    }

    #[test]
    fn should_read_a_module_summary_up_to_the_first_code_line() {
        assert_eq!(
            module_summary_of("//! Summary.\n//! More.\n\nuse a;\n//! Late.\n"),
            "Summary. More."
        );
    }

    #[test]
    fn should_recognize_an_identifier_only_as_a_whole_word() {
        assert!(mentions_whole_word("raw::mlx_add", "mlx_add"));
        assert!(!mentions_whole_word("raw::mlx_addmm(x)", "mlx_add"));
        assert!(!mentions_whole_word("", "mlx_add"));
    }
}
