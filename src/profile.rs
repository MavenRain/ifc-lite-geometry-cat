//! 2D profile definitions for extrusion.
//!
//! A [`Profile2D`] describes the cross-section of a building element
//! (wall, column, beam, etc.) that is later extruded into a 3D
//! [`Mesh`](crate::mesh::Mesh).

use crate::error::{Error, Result};
use nalgebra::Point2;

// ═══════════════════════════════════════════════════════════════════
// Profile2D
// ═══════════════════════════════════════════════════════════════════

/// A 2D profile with an outer boundary and optional holes.
///
/// The outer boundary should be counter-clockwise; holes should be
/// clockwise.  All fields are private.
///
/// # Examples
///
/// ```
/// use ifc_lite_geometry_cat::Profile2D;
///
/// let rect = Profile2D::rectangle(10.0, 5.0);
/// assert_eq!(rect.outer().len(), 4);
/// assert!(rect.holes().is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct Profile2D {
    outer: Vec<Point2<f64>>,
    holes: Vec<Vec<Point2<f64>>>,
}

impl Profile2D {
    /// Create from an outer boundary with no holes.
    #[must_use]
    pub fn new(outer: Vec<Point2<f64>>) -> Self {
        Self {
            outer,
            holes: Vec::new(),
        }
    }

    /// Create from an outer boundary and a set of holes.
    #[must_use]
    pub fn with_holes(outer: Vec<Point2<f64>>, holes: Vec<Vec<Point2<f64>>>) -> Self {
        Self { outer, holes }
    }

    /// Outer boundary vertices.
    #[must_use]
    pub fn outer(&self) -> &[Point2<f64>] {
        &self.outer
    }

    /// Hole contours.
    #[must_use]
    pub fn holes(&self) -> &[Vec<Point2<f64>>] {
        &self.holes
    }

    // ── Named constructors ──────────────────────────────────────────

    /// A rectangular profile centred at the origin.
    #[must_use]
    pub fn rectangle(width: f64, height: f64) -> Self {
        let hw = width / 2.0;
        let hh = height / 2.0;
        Self::new(vec![
            Point2::new(-hw, -hh),
            Point2::new(hw, -hh),
            Point2::new(hw, hh),
            Point2::new(-hw, hh),
        ])
    }

    /// A circular profile centred at the origin.
    #[must_use]
    pub fn circle(radius: f64) -> Self {
        let segments = circle_segments(radius);
        let outer = (0..segments)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let angle =
                    2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
                Point2::new(radius * angle.cos(), radius * angle.sin())
            })
            .collect();
        Self::new(outer)
    }

    /// A hollow circular profile (pipe section).
    #[must_use]
    pub fn hollow_circle(outer_radius: f64, inner_radius: f64) -> Self {
        let outer_segments = circle_segments(outer_radius);
        let inner_segments = circle_segments(inner_radius);

        let outer: Vec<_> = (0..outer_segments)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let angle = 2.0 * std::f64::consts::PI * (i as f64)
                    / (outer_segments as f64);
                Point2::new(outer_radius * angle.cos(), outer_radius * angle.sin())
            })
            .collect();

        let hole: Vec<_> = (0..inner_segments)
            .rev()
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let angle = 2.0 * std::f64::consts::PI * (i as f64)
                    / (inner_segments as f64);
                Point2::new(inner_radius * angle.cos(), inner_radius * angle.sin())
            })
            .collect();

        Self::with_holes(outer, vec![hole])
    }

    // ── Triangulation ───────────────────────────────────────────────

    /// Triangulate this profile using `earcutr`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidProfile`] when the outer boundary has
    /// fewer than 3 vertices, or [`Error::Triangulation`] on earcut
    /// failure.
    pub fn triangulate(&self) -> Result<Triangulation> {
        (self.outer.len() >= 3).then_some(()).ok_or_else(|| {
            Error::InvalidProfile("profile must have at least 3 vertices".into())
        })?;

        // Flatten outer + holes into a single coordinate array.
        let hole_point_count: usize = self.holes.iter().map(Vec::len).sum();
        let total = self.outer.len() + hole_point_count;

        let vertices: Vec<f64> = self
            .outer
            .iter()
            .chain(self.holes.iter().flatten())
            .flat_map(|p| [p.x, p.y])
            .collect();

        // Build hole-start indices (in vertex count, not coordinate count).
        let hole_indices: Vec<usize> = self
            .holes
            .iter()
            .scan(self.outer.len(), |start, hole| {
                let idx = *start;
                *start += hole.len();
                Some(idx)
            })
            .collect();

        let tri_indices = if hole_indices.is_empty() {
            earcutr::earcut(&vertices, &[], 2)
        } else {
            earcutr::earcut(&vertices, &hole_indices, 2)
        }
        .map_err(|e| Error::Triangulation(format!("{e:?}")))?;

        let points: Vec<Point2<f64>> = vertices
            .chunks_exact(2)
            .map(|c| Point2::new(c[0], c[1]))
            .collect();

        debug_assert_eq!(points.len(), total);

        Ok(Triangulation {
            points,
            indices: tri_indices,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Triangulation
// ═══════════════════════════════════════════════════════════════════

/// Result of triangulating a [`Profile2D`].
#[derive(Debug, Clone)]
pub struct Triangulation {
    points: Vec<Point2<f64>>,
    indices: Vec<usize>,
}

impl Triangulation {
    /// All vertices (outer + holes combined).
    #[must_use]
    pub fn points(&self) -> &[Point2<f64>] {
        &self.points
    }

    /// Triangle indices into [`points`](Self::points).
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

// ═══════════════════════════════════════════════════════════════════
// ProfileType
// ═══════════════════════════════════════════════════════════════════

/// Named profile types that can be converted to [`Profile2D`].
#[derive(Debug, Clone)]
pub enum ProfileType {
    /// Centred rectangle.
    Rectangle { width: f64, height: f64 },
    /// Solid circle.
    Circle { radius: f64 },
    /// Hollow circle (pipe).
    HollowCircle {
        outer_radius: f64,
        inner_radius: f64,
    },
    /// Arbitrary polygon.
    Polygon { points: Vec<Point2<f64>> },
}

impl ProfileType {
    /// Convert to a [`Profile2D`].
    #[must_use]
    pub fn to_profile(&self) -> Profile2D {
        match self {
            Self::Rectangle { width, height } => Profile2D::rectangle(*width, *height),
            Self::Circle { radius } => Profile2D::circle(*radius),
            Self::HollowCircle {
                outer_radius,
                inner_radius,
            } => Profile2D::hollow_circle(*outer_radius, *inner_radius),
            Self::Polygon { points } => Profile2D::new(points.clone()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

/// Adaptive segment count for circle tessellation.
#[must_use]
fn circle_segments(radius: f64) -> usize {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let segments = (radius.sqrt() * 8.0).ceil() as usize;
    segments.clamp(8, 32)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_profile() {
        let p = Profile2D::rectangle(10.0, 5.0);
        assert_eq!(p.outer().len(), 4);
        assert!(p.holes().is_empty());
        assert_eq!(p.outer()[0], Point2::new(-5.0, -2.5));
    }

    #[test]
    fn circle_profile_on_circle() {
        let p = Profile2D::circle(5.0);
        assert!(p.outer().len() >= 8);
        let first = p.outer()[0];
        let dist = (first.x * first.x + first.y * first.y).sqrt();
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn hollow_circle_has_hole() {
        let p = Profile2D::hollow_circle(10.0, 5.0);
        assert!(p.outer().len() >= 8);
        assert_eq!(p.holes().len(), 1);
    }

    #[test]
    fn triangulate_rectangle() {
        let p = Profile2D::rectangle(10.0, 5.0);
        let tri = p.triangulate().expect("triangulate");
        assert_eq!(tri.points().len(), 4);
        assert_eq!(tri.indices().len(), 6); // 2 triangles
    }

    #[test]
    fn triangulate_circle() {
        let p = Profile2D::circle(5.0);
        let tri = p.triangulate().expect("triangulate");
        assert!(tri.points().len() >= 8);
        assert_eq!(tri.indices().len(), (tri.points().len() - 2) * 3);
    }

    #[test]
    fn triangulate_hollow_circle() {
        let p = Profile2D::hollow_circle(10.0, 5.0);
        let tri = p.triangulate().expect("triangulate");
        let expected_points = circle_segments(10.0) + circle_segments(5.0);
        assert_eq!(tri.points().len(), expected_points);
    }

    #[test]
    fn profile_type_round_trip() {
        let pt = ProfileType::Rectangle {
            width: 4.0,
            height: 2.0,
        };
        let p = pt.to_profile();
        assert_eq!(p.outer().len(), 4);
    }
}
