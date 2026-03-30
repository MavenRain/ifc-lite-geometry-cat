//! # IFC-Lite Geometry Processing
//!
//! Geometry processing and mesh generation for IFC models, built on
//! [`comp_cat_rs`].
//!
//! ## Overview
//!
//! This crate transforms IFC geometry representations into GPU-ready
//! triangle meshes:
//!
//! - **Profile Handling** -- 2D profiles (rectangle, circle, arbitrary)
//! - **Extrusion** -- 3D meshes from extruded profiles via
//!   [`comp_cat_rs::effect::io::Io`]
//! - **Triangulation** -- polygon triangulation with hole support
//! - **Mesh Processing** -- merging, bounds, coordinate shifts
//!
//! ## Quick Start
//!
//! ```rust
//! use ifc_lite_geometry_cat::{Profile2D, extrude_profile};
//!
//! let profile = Profile2D::rectangle(2.0, 1.0);
//! let mesh = extrude_profile(&profile, 3.0, None).run().unwrap();
//! assert!(mesh.triangle_count() > 0);
//! ```
//!
//! ## Functional Design
//!
//! All geometry operations are pure functions.  Effectful operations
//! like extrusion return [`Io<Error, Mesh>`](comp_cat_rs::effect::io::Io)
//! so they compose via `map` and `flat_map`.  Call `run` only at the
//! outermost boundary.

pub mod error;
pub mod extrusion;
pub mod mesh;
pub mod profile;
pub mod triangulation;

// ── Convenience re-exports ──────────────────────────────────────────

pub use error::{Error, Result};
pub use extrusion::{apply_transform, extrude_profile};
pub use mesh::{CoordinateShift, Mesh};
pub use nalgebra::{Point2, Point3, Vector2, Vector3};
pub use profile::{Profile2D, ProfileType, Triangulation};
pub use triangulation::{
    calculate_polygon_normal, project_to_2d, triangulate_polygon,
    triangulate_polygon_with_holes,
};
