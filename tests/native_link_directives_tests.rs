//! Contracts for the Cargo link directives composed for a prefix that holds
//! the MLX-C library, and for the reader that recovers the libraries MLX was
//! configured against from MLX's installed CMake package. The module is a
//! source file shared with the build script, so these tests include it
//! directly, the way the digest contracts include the engine. Nothing here
//! needs a native build.

#[path = "../native_link_directives.rs"]
mod native_link_directives;

use std::path::{Path, PathBuf};

use native_link_directives::{
    ExportedLinkLibrary, MLX_EXPORTED_TARGETS_FILE, StaticPrefixArchives,
    exported_mlx_link_libraries, linux_static_pair_link_directives,
    macos_static_pair_link_directives, shared_pair_link_directives,
};

const COMPLETE_PREFIX: StaticPrefixArchives = StaticPrefixArchives {
    carries_gap_shim: true,
    carries_gguf_archive: true,
};
const BARE_PREFIX: StaticPrefixArchives = StaticPrefixArchives {
    carries_gap_shim: false,
    carries_gguf_archive: false,
};

/// The `mlx` target as a Linux install of MLX exports it: private
/// dependencies wrapped in LINK_ONLY, the BLAS CMake found repeated once per
/// find_package that returned it, and empty wrappers where a variable
/// expanded to nothing.
const LINUX_EXPORTED_TARGETS: &str = r#"
add_library(mlx STATIC IMPORTED)

set_target_properties(mlx PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "MLX_USE_ACCELERATE=0"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "\$<LINK_ONLY:dl>;\$<LINK_ONLY:pthread>;/opt/blas/lib/libopenblas.so;/opt/blas/lib/libopenblas.so;\$<LINK_ONLY:>;\$<LINK_ONLY:>"
)
"#;

fn position_of(link_directives: &[String], expected_directive: &str) -> usize {
    link_directives
        .iter()
        .position(|link_directive| link_directive == expected_directive)
        .unwrap_or_else(|| panic!("missing directive {expected_directive} in {link_directives:?}"))
}

#[test]
fn should_read_each_exported_library_once_in_order() {
    let exported_link_libraries = exported_mlx_link_libraries(LINUX_EXPORTED_TARGETS)
        .expect("the mlx target names its link libraries");

    assert_eq!(
        exported_link_libraries,
        vec![
            ExportedLinkLibrary::LibraryName("dl".to_owned()),
            ExportedLinkLibrary::LibraryName("pthread".to_owned()),
            ExportedLinkLibrary::LibraryFile(PathBuf::from("/opt/blas/lib/libopenblas.so")),
        ]
    );
}

#[test]
fn should_skip_imported_targets_and_generator_expressions() {
    // The macOS shape: jaccl is an imported target and the frameworks are
    // generator expressions, neither of which a linker can be handed.
    let exported_targets = r#"
set_target_properties(mlx PROPERTIES
  INTERFACE_LINK_LIBRARIES "jaccl::jaccl;\$<LINK_ONLY:$<TARGET_NAME:fmt>>;-lm"
)
"#;

    assert_eq!(
        exported_mlx_link_libraries(exported_targets),
        Some(vec![ExportedLinkLibrary::LibraryName("m".to_owned())])
    );
}

#[test]
fn should_report_no_libraries_when_the_property_is_empty() {
    // Boundary: an MLX with nothing to link still has a readable package.
    let exported_targets = "set_target_properties(mlx PROPERTIES\n  \
                            INTERFACE_LINK_LIBRARIES \"\"\n)\n";

    assert_eq!(exported_mlx_link_libraries(exported_targets), Some(vec![]));
}

#[test]
fn should_refuse_a_file_that_does_not_describe_the_mlx_target() {
    assert_eq!(exported_mlx_link_libraries(""), None);
    assert_eq!(
        exported_mlx_link_libraries(
            "set_target_properties(other PROPERTIES\n  \
             INTERFACE_LINK_LIBRARIES \"/opt/blas/lib/libopenblas.so\"\n)\n"
        ),
        None
    );
}

#[test]
fn should_not_take_link_libraries_from_a_later_target() {
    // The mlx target has no link libraries here; the property that follows
    // belongs to another target and must not be attributed to mlx.
    let exported_targets = "set_target_properties(mlx PROPERTIES\n  \
                            INTERFACE_INCLUDE_DIRECTORIES \"include\"\n)\n\n\
                            set_target_properties(other PROPERTIES\n  \
                            INTERFACE_LINK_LIBRARIES \"/opt/blas/lib/libopenblas.so\"\n)\n";

    assert_eq!(exported_mlx_link_libraries(exported_targets), None);
}

#[test]
fn should_link_the_linux_archives_before_the_libraries_they_reference() {
    let exported_link_libraries = exported_mlx_link_libraries(LINUX_EXPORTED_TARGETS)
        .expect("the mlx target names its link libraries");

    let link_directives =
        linux_static_pair_link_directives(COMPLETE_PREFIX, &exported_link_libraries);

    // The linker resolves archives left to right.
    let gap_shim_position = position_of(
        &link_directives,
        "cargo:rustc-link-lib=static=mlxrs_gap_shim",
    );
    let mlxc_position = position_of(&link_directives, "cargo:rustc-link-lib=static=mlxc");
    let mlx_position = position_of(&link_directives, "cargo:rustc-link-lib=static=mlx");
    let gguf_position = position_of(&link_directives, "cargo:rustc-link-lib=static=gguflib");
    let blas_position = position_of(&link_directives, "cargo:rustc-link-lib=dylib=openblas");
    let runtime_position = position_of(&link_directives, "cargo:rustc-link-lib=dylib=stdc++");
    assert!(gap_shim_position < mlxc_position);
    assert!(mlxc_position < mlx_position);
    assert!(mlx_position < gguf_position);
    assert!(gguf_position < blas_position);
    assert!(blas_position < runtime_position);
}

#[test]
fn should_put_a_resolved_library_directory_on_the_search_and_run_paths() {
    let blas_file = ExportedLinkLibrary::LibraryFile(PathBuf::from("/opt/blas/lib/libopenblas.so"));

    let link_directives = linux_static_pair_link_directives(BARE_PREFIX, &[blas_file]);

    position_of(
        &link_directives,
        "cargo:rustc-link-search=native=/opt/blas/lib",
    );
    position_of(
        &link_directives,
        "cargo:rustc-link-arg=-Wl,-rpath,/opt/blas/lib",
    );
}

#[test]
fn should_link_a_resolved_archive_statically_and_warn_about_its_runtime() {
    let blas_archive =
        ExportedLinkLibrary::LibraryFile(PathBuf::from("/opt/blas/lib/libopenblas.a"));

    let link_directives = linux_static_pair_link_directives(BARE_PREFIX, &[blas_archive]);

    position_of(&link_directives, "cargo:rustc-link-lib=static=openblas");
    assert!(
        !link_directives
            .iter()
            .any(|link_directive| link_directive.contains("-rpath")),
        "an archive is embedded and needs no run path: {link_directives:?}"
    );
    // A static BLAS carries undeclared runtime dependencies; the consumer is
    // told to link them instead of meeting undefined symbols unexplained.
    assert!(
        link_directives.iter().any(
            |link_directive| link_directive.starts_with("cargo:warning=")
                && link_directive.contains("libopenblas.a")
        ),
        "the static archive must be named in a warning: {link_directives:?}"
    );
}

#[test]
fn should_report_linker_flags_in_the_package_as_unsupported_entries() {
    let exported_targets = r#"
set_target_properties(mlx PROPERTIES
  INTERFACE_LINK_LIBRARIES "\$<LINK_ONLY:-pthread>;-fopenmp;-Wl,--as-needed;-l:libblas.so.3;-lgomp;-l"
)
"#;

    assert_eq!(
        exported_mlx_link_libraries(exported_targets),
        Some(vec![
            ExportedLinkLibrary::UnsupportedEntry("-pthread".to_owned()),
            ExportedLinkLibrary::UnsupportedEntry("-fopenmp".to_owned()),
            ExportedLinkLibrary::UnsupportedEntry("-Wl,--as-needed".to_owned()),
            ExportedLinkLibrary::UnsupportedEntry("-l:libblas.so.3".to_owned()),
            ExportedLinkLibrary::LibraryName("gomp".to_owned()),
            ExportedLinkLibrary::UnsupportedEntry("-l".to_owned()),
        ])
    );
}

#[test]
fn should_warn_about_an_unsupported_entry_instead_of_linking_it() {
    let link_directives = linux_static_pair_link_directives(
        BARE_PREFIX,
        &[ExportedLinkLibrary::UnsupportedEntry("-pthread".to_owned())],
    );

    assert!(
        link_directives.iter().any(
            |link_directive| link_directive.starts_with("cargo:warning=")
                && link_directive.contains("-pthread")
        ),
        "the flag must be named in a warning: {link_directives:?}"
    );
    assert!(
        !link_directives
            .iter()
            .any(|link_directive| link_directive.contains("rustc-link-lib")
                && link_directive.contains("pthread")),
        "a flag must never become a library name: {link_directives:?}"
    );
}

#[test]
fn should_warn_instead_of_linking_a_library_file_the_linker_cannot_name() {
    // A versioned shared library has no `-l` spelling.
    let versioned_library =
        ExportedLinkLibrary::LibraryFile(PathBuf::from("/opt/blas/lib/libopenblas.so.0"));

    let link_directives = linux_static_pair_link_directives(BARE_PREFIX, &[versioned_library]);

    assert!(
        link_directives.iter().any(
            |link_directive| link_directive.starts_with("cargo:warning=")
                && link_directive.contains("libopenblas.so.0")
        ),
        "the unlinkable library must be named in a warning: {link_directives:?}"
    );
    assert!(
        !link_directives
            .iter()
            .any(|link_directive| link_directive.contains("rustc-link-lib")
                && link_directive.contains("openblas")),
    );
}

#[test]
fn should_link_only_the_pair_and_the_runtime_for_a_bare_linux_prefix() {
    // Boundary: a consumer-supplied prefix with neither optional archive and
    // an MLX package that names no libraries.
    assert_eq!(
        linux_static_pair_link_directives(BARE_PREFIX, &[]),
        vec![
            "cargo:rustc-link-lib=static=mlxc",
            "cargo:rustc-link-lib=static=mlx",
            "cargo:rustc-link-lib=dylib=stdc++",
        ]
    );
}

#[test]
fn should_keep_the_macos_static_link_line_unchanged() {
    // The line this crate has always emitted for the engine's own prefix on
    // macOS; the Linux support must not move a single entry of it.
    assert_eq!(
        macos_static_pair_link_directives(COMPLETE_PREFIX),
        vec![
            "cargo:rustc-link-lib=static=mlxrs_gap_shim",
            "cargo:rustc-link-lib=static=mlxc",
            "cargo:rustc-link-lib=static=mlx",
            "cargo:rustc-link-lib=static=gguflib",
            "cargo:rustc-link-lib=static=jaccl",
            "cargo:rustc-link-lib=framework=Metal",
            "cargo:rustc-link-lib=framework=Foundation",
            "cargo:rustc-link-lib=framework=QuartzCore",
            "cargo:rustc-link-lib=framework=Accelerate",
            "cargo:rustc-link-lib=dylib=c++",
        ]
    );
}

#[test]
fn should_omit_the_optional_archives_a_macos_prefix_does_not_carry() {
    let link_directives = macos_static_pair_link_directives(BARE_PREFIX);

    assert!(
        !link_directives
            .iter()
            .any(|link_directive| link_directive.contains("mlxrs_gap_shim")
                || link_directive.contains("gguflib")),
        "absent archives must not be linked: {link_directives:?}"
    );
}

#[test]
fn should_link_a_shared_pair_with_a_run_path_to_its_directory() {
    assert_eq!(
        shared_pair_link_directives(Path::new("/opt/mlx/lib")),
        vec![
            "cargo:rustc-link-lib=dylib=mlxc",
            "cargo:rustc-link-lib=dylib=mlx",
            "cargo:rustc-link-arg=-Wl,-rpath,/opt/mlx/lib",
        ]
    );
}

#[test]
fn should_locate_the_exported_targets_file_inside_the_prefix() {
    // The build script joins this onto the prefix, and joining an absolute
    // path would discard the prefix and read a file outside it.
    assert!(Path::new(MLX_EXPORTED_TARGETS_FILE).is_relative());
    assert!(MLX_EXPORTED_TARGETS_FILE.ends_with("MLXTargets.cmake"));
}
