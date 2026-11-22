//! The Cargo link directives for a prefix that holds the MLX-C library, as
//! pure functions over what the prefix carries, so the link line for every
//! host is covered by contracts that need no native build. A self-contained,
//! std-only source module shared by the build script and the test tree.
//!
//! Every directive is `rustc-link-lib` or `rustc-link-search`, which Cargo
//! passes on to the crates that depend on this one. The rpath arguments are
//! the exception: `rustc-link-arg` reaches only this crate's own tests and
//! examples.

use std::path::{Path, PathBuf};

/// The optional archives beside the static pair in a prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticPrefixArchives {
    /// The repository's gap shim. Only the engine's prefix carries it; a
    /// consumer-supplied prefix omits it and the mlxrs_* wrappers stay
    /// unlinked unless the consumer compiles the shim themselves.
    pub carries_gap_shim: bool,
    /// MLX's GGUF archive, which MLX makes optional.
    pub carries_gguf_archive: bool,
}

/// One library MLX's installed CMake package says a static consumer must
/// link, as CMake resolved it when MLX was configured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportedLinkLibrary {
    /// A library CMake resolved to a file, such as the BLAS it found.
    LibraryFile(PathBuf),
    /// A library named for the linker's own search, such as `pthread`.
    LibraryName(String),
    /// An entry this crate cannot hand to Cargo as a library, such as the
    /// compiler flag `-pthread` or the exact-file form `-l:libfoo.so.3`. It
    /// is reported in a warning instead of being mistaken for a name.
    UnsupportedEntry(String),
}

/// Directives for a prefix whose `lib` holds shared libraries: the two
/// libraries plus an rpath to their directory, because their install names
/// are relative to the loader's search path.
pub fn shared_pair_link_directives(library_directory: &Path) -> Vec<String> {
    vec![
        "cargo:rustc-link-lib=dylib=mlxc".to_owned(),
        "cargo:rustc-link-lib=dylib=mlx".to_owned(),
        format!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            library_directory.display()
        ),
    ]
}

/// The static archives every host links, in resolution order. Order matters
/// for archives, which the linker resolves left to right: the gap shim
/// references mlxc and mlx, mlxc references mlx, and mlx references GGUF.
fn static_pair_archive_directives(prefix_archives: StaticPrefixArchives) -> Vec<String> {
    let mut link_directives = Vec::new();
    if prefix_archives.carries_gap_shim {
        link_directives.push("cargo:rustc-link-lib=static=mlxrs_gap_shim".to_owned());
    }
    link_directives.push("cargo:rustc-link-lib=static=mlxc".to_owned());
    link_directives.push("cargo:rustc-link-lib=static=mlx".to_owned());
    if prefix_archives.carries_gguf_archive {
        link_directives.push("cargo:rustc-link-lib=static=gguflib".to_owned());
    }
    link_directives
}

/// Directives for a static prefix on macOS: the archives, the jaccl archive
/// MLX links as its own library there, the Apple frameworks, and the C++
/// runtime the shared libraries used to carry transitively.
pub fn macos_static_pair_link_directives(prefix_archives: StaticPrefixArchives) -> Vec<String> {
    let mut link_directives = static_pair_archive_directives(prefix_archives);
    // Follows mlx, which references it.
    link_directives.push("cargo:rustc-link-lib=static=jaccl".to_owned());
    for framework_name in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
        link_directives.push(format!("cargo:rustc-link-lib=framework={framework_name}"));
    }
    link_directives.push("cargo:rustc-link-lib=dylib=c++".to_owned());
    link_directives
}

/// Directives for a static prefix on Linux: the archives, then the libraries
/// MLX's own package records (the BLAS and LAPACK CMake found when MLX was
/// configured), then the GNU C++ runtime.
///
/// The BLAS is taken from the prefix instead of being named here because
/// Linux has several (OpenBLAS, the reference BLAS, vendor builds) in
/// distribution-specific places; MLX's configure already chose one, and
/// linking any other would mismatch the archive that was compiled.
pub fn linux_static_pair_link_directives(
    prefix_archives: StaticPrefixArchives,
    exported_link_libraries: &[ExportedLinkLibrary],
) -> Vec<String> {
    let mut link_directives = static_pair_archive_directives(prefix_archives);
    for exported_link_library in exported_link_libraries {
        match exported_link_library {
            ExportedLinkLibrary::LibraryName(library_name) => {
                link_directives.push(format!("cargo:rustc-link-lib=dylib={library_name}"));
            }
            ExportedLinkLibrary::LibraryFile(library_file) => {
                link_directives.extend(library_file_link_directives(library_file));
            }
            ExportedLinkLibrary::UnsupportedEntry(link_entry) => {
                link_directives.push(format!(
                    "cargo:warning=MLX's package asks static consumers to link with `{link_entry}`, \
                     which is not a library this crate can name to Cargo; it is not passed on, \
                     and a binary that needs it must add it to its own link line"
                ));
            }
        }
    }
    link_directives.push("cargo:rustc-link-lib=dylib=stdc++".to_owned());
    link_directives
}

/// Directives for one library CMake resolved to a file: its directory on the
/// search path, the library by name, and an rpath so this crate's own tests
/// load a shared library that lives outside the loader's default directories.
/// Cargo passes that rpath to nothing else, so a dependent binary loading a
/// shared library from such a directory needs its own rpath or the loader's
/// search path; the Linux section of docs/native-build.md says so. A static
/// archive links, but its own runtime dependencies (a Fortran or OpenMP
/// runtime for a static BLAS) are not recorded beside it, so they are named
/// in a warning. A file that does not follow the `lib<name>.so` or
/// `lib<name>.a` convention cannot be named to the linker and yields a
/// warning instead.
fn library_file_link_directives(library_file: &Path) -> Vec<String> {
    let Some((library_kind, library_name)) = linker_kind_and_name(library_file) else {
        return vec![format!(
            "cargo:warning=MLX's package names the library {} in a form the linker \
             cannot be given by name; it is not linked",
            library_file.display()
        )];
    };
    let mut link_directives = Vec::new();
    if let Some(library_directory) = library_file.parent() {
        link_directives.push(format!(
            "cargo:rustc-link-search=native={}",
            library_directory.display()
        ));
        if library_kind == "dylib" {
            link_directives.push(format!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                library_directory.display()
            ));
        }
    }
    link_directives.push(format!(
        "cargo:rustc-link-lib={library_kind}={library_name}"
    ));
    if library_kind == "static" {
        link_directives.push(format!(
            "cargo:warning=MLX was configured against the static library {}; the \
             libraries it depends on itself (for a static BLAS, typically gfortran, gomp \
             or pthread) are not recorded and must be linked by the consuming binary",
            library_file.display()
        ));
    }
    link_directives
}

/// The Cargo link kind and the linker name of a library file: `dylib` for
/// `lib<name>.so`, `static` for `lib<name>.a`, and `None` for any other
/// file name (a versioned `lib<name>.so.3`, for example, has no `-l` form).
fn linker_kind_and_name(library_file: &Path) -> Option<(&'static str, String)> {
    let file_name = library_file.file_name()?.to_str()?;
    let unprefixed_name = file_name.strip_prefix("lib")?;
    if let Some(shared_name) = unprefixed_name.strip_suffix(".so") {
        return Some(("dylib", shared_name.to_owned()));
    }
    let archive_name = unprefixed_name.strip_suffix(".a")?;
    Some(("static", archive_name.to_owned()))
}

/// Where an installed MLX keeps the exported targets of its CMake package,
/// relative to the prefix.
pub const MLX_EXPORTED_TARGETS_FILE: &str = "share/cmake/MLX/MLXTargets.cmake";

/// Reads the libraries a static consumer of MLX must link from the content
/// of MLX's exported targets file: the `INTERFACE_LINK_LIBRARIES` of the
/// `mlx` target, each entry listed once, in order. Returns `None` when the
/// file names no such property, which means the prefix was not produced by
/// an MLX install this reader understands.
pub fn exported_mlx_link_libraries(
    exported_targets_content: &str,
) -> Option<Vec<ExportedLinkLibrary>> {
    const MLX_TARGET_PROPERTIES: &str = "set_target_properties(mlx PROPERTIES";
    const LINK_LIBRARIES_PROPERTY: &str = "INTERFACE_LINK_LIBRARIES \"";
    let (_, mlx_target_properties) = exported_targets_content.split_once(MLX_TARGET_PROPERTIES)?;
    let (mlx_target_properties, _) = mlx_target_properties.split_once("\n)")?;
    let (_, link_libraries_value) = mlx_target_properties.split_once(LINK_LIBRARIES_PROPERTY)?;
    let (link_libraries_value, _) = link_libraries_value.split_once('"')?;

    let mut exported_link_libraries = Vec::new();
    for link_library_entry in link_libraries_value.split(';') {
        let Some(exported_link_library) = exported_link_library(link_library_entry) else {
            continue;
        };
        if !exported_link_libraries.contains(&exported_link_library) {
            exported_link_libraries.push(exported_link_library);
        }
    }
    Some(exported_link_libraries)
}

/// One entry of the property as a library to link, or `None` for an entry
/// a linker cannot use.
fn exported_link_library(link_library_entry: &str) -> Option<ExportedLinkLibrary> {
    // CMake wraps a private dependency of a static library as
    // $<LINK_ONLY:name>, escaping the dollar sign in the file.
    let link_library_entry = match link_library_entry.strip_prefix("\\$<LINK_ONLY:") {
        Some(wrapped_entry) => wrapped_entry.strip_suffix('>')?,
        None => link_library_entry,
    }
    .trim();
    // Imported targets (name::name) and remaining generator expressions are
    // resolved by CMake, not by a linker; nothing on Linux uses them for the
    // libraries this reader exists to find.
    if link_library_entry.is_empty()
        || link_library_entry.contains("::")
        || link_library_entry.contains("$<")
    {
        return None;
    }
    if Path::new(link_library_entry).is_absolute() {
        return Some(ExportedLinkLibrary::LibraryFile(PathBuf::from(
            link_library_entry,
        )));
    }
    // A bare name, or `-l<name>`, is a library the linker finds by name.
    // Anything else starting with a dash is a flag (`-pthread`, `-fopenmp`,
    // `-Wl,...`) or the exact-file form `-l:<file>`, neither of which Cargo
    // accepts as a library name.
    let library_name = link_library_entry
        .strip_prefix("-l")
        .unwrap_or(link_library_entry);
    if library_name.is_empty() || library_name.starts_with('-') || library_name.contains(':') {
        return Some(ExportedLinkLibrary::UnsupportedEntry(
            link_library_entry.to_owned(),
        ));
    }
    Some(ExportedLinkLibrary::LibraryName(library_name.to_owned()))
}
