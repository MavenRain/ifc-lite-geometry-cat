//! Polygon triangulation utilities.
//!
//! Provides functions for triangulating simple and holed polygons,
//! projecting 3D points onto 2D planes, and computing polygon
//! normals.  All functions are pure.

use crate::error::{Error, Result};
use nalgebra::{Point2, Point3, Vector3};

/// Triangulate a simple polygon (no holes).
///
/// Returns triangle indices into the input `points` slice.
///
/// # Errors
///
/// Returns [`Error::Triangulation`] when `points.len() < 3` or when
/// earcut fails.
pub fn triangulate_polygon(points: &[Point2<f64>]) -> Result<Vec<usize>> {
    let n = points.len();

    (n >= 3).then_some(()).ok_or_else(|| {
        Error::Triangulation("need at least 3 points to triangulate".into())
    })?;

    // Fast paths.
    match n {
        3 => Ok(vec![0, 1, 2]),
        4 if is_convex(points) => Ok(vec![0, 1, 2, 0, 2, 3]),
        _ if n <= 8 && is_convex(points) => Ok(fan_triangulate(n)),
        _ => {
            let vertices: Vec<f64> = points.iter().flat_map(|p| [p.x, p.y]).collect();
            earcutr::earcut(&vertices, &[], 2)
                .map_err(|e| Error::Triangulation(format!("{e:?}")))
        }
    }
}

/// Triangulate a polygon with holes.
///
/// Returns indices into the concatenated vertex array
/// (outer + all holes in order).
///
/// # Errors
///
/// Returns [`Error::Triangulation`] on failure.
pub fn triangulate_polygon_with_holes(
    outer: &[Point2<f64>],
    holes: &[Vec<Point2<f64>>],
) -> Result<Vec<usize>> {
    (outer.len() >= 3).then_some(()).ok_or_else(|| {
        Error::Triangulation("need at least 3 points in outer boundary".into())
    })?;

    let valid_holes: Vec<&Vec<Point2<f64>>> =
        holes.iter().filter(|h| h.len() >= 3).collect();

    if valid_holes.is_empty() {
        triangulate_polygon(outer)
    } else {
        let vertices: Vec<f64> = outer
            .iter()
            .chain(valid_holes.iter().flat_map(|h| h.iter()))
            .flat_map(|p| [p.x, p.y])
            .collect();

        let hole_indices: Vec<usize> = valid_holes
            .iter()
            .scan(outer.len(), |start, hole| {
                let idx = *start;
                *start += hole.len();
                Some(idx)
            })
            .collect();

        earcutr::earcut(&vertices, &hole_indices, 2)
            .map_err(|e| Error::Triangulation(format!("{e:?}")))
    }
}

/// Project 3D points onto a 2D plane defined by `normal`.
///
/// Returns `(points_2d, u_axis, v_axis, origin)`.
#[must_use]
pub fn project_to_2d(
    points_3d: &[Point3<f64>],
    normal: &Vector3<f64>,
) -> (Vec<Point2<f64>>, Vector3<f64>, Vector3<f64>, Point3<f64>) {
    points_3d.first().map_or_else(
        || (Vec::new(), Vector3::zeros(), Vector3::zeros(), Point3::origin()),
        |&origin| {
            let reference = least_parallel_axis(normal);
            let u_axis = normal.cross(&reference).normalize();
            let v_axis = normal.cross(&u_axis).normalize();

            let pts = points_3d
                .iter()
                .map(|p| {
                    let v = p - origin;
                    Point2::new(v.dot(&u_axis), v.dot(&v_axis))
                })
                .collect();

            (pts, u_axis, v_axis, origin)
        },
    )
}

/// Calculate the normal of a polygon using Newell's method.
#[must_use]
pub fn calculate_polygon_normal(points: &[Point3<f64>]) -> Vector3<f64> {
    match points.len() {
        0..=2 => Vector3::z(),
        3 | 4 => {
            let v1 = points[1] - points[0];
            let v2 = points[2] - points[0];
            v1.cross(&v2)
                .try_normalize(1e-10)
                .or_else(|| {
                    points.get(3).and_then(|p3| {
                        let v3 = p3 - points[0];
                        v2.cross(&v3).try_normalize(1e-10)
                    })
                })
                .unwrap_or_else(Vector3::z)
        }
        _ => {
            // Newell's method for robust normal on complex polygons.
            let normal = points
                .iter()
                .zip(points.iter().cycle().skip(1))
                .fold(Vector3::zeros(), |acc, (cur, next)| {
                    Vector3::new(
                        acc.x + (cur.y - next.y) * (cur.z + next.z),
                        acc.y + (cur.z - next.z) * (cur.x + next.x),
                        acc.z + (cur.x - next.x) * (cur.y + next.y),
                    )
                });
            normal.try_normalize(1e-10).unwrap_or_else(Vector3::z)
        }
    }
}

// ── helpers ─────────────────────────────────────────────────────────

fn is_convex(points: &[Point2<f64>]) -> bool {
    let n = points.len();
    (n >= 3)
        && (0..n)
            .map(|i| {
                let p0 = &points[i];
                let p1 = &points[(i + 1) % n];
                let p2 = &points[(i + 2) % n];
                (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x)
            })
            .try_fold(0i8, |sign, cross| {
                match (cross.abs() > 1e-10, sign) {
                    (false, _) => Some(sign),
                    (true, 0) => Some(if cross > 0.0 { 1 } else { -1 }),
                    (true, s) => {
                        let current = if cross > 0.0 { 1 } else { -1 };
                        (s == current).then_some(s)
                    }
                }
            })
            .is_some()
}

fn fan_triangulate(n: usize) -> Vec<usize> {
    (1..n - 1).flat_map(|i| [0, i, i + 1]).collect()
}

fn least_parallel_axis(normal: &Vector3<f64>) -> Vector3<f64> {
    let ax = normal.x.abs();
    let ay = normal.y.abs();
    let az = normal.z.abs();
    match () {
        () if ax <= ay && ax <= az => Vector3::x(),
        () if ay <= az => Vector3::y(),
        () => Vector3::z(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulate_square() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let idx = triangulate_polygon(&pts).expect("tri");
        assert_eq!(idx.len(), 6);
    }

    #[test]
    fn triangulate_triangle() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        assert_eq!(triangulate_polygon(&pts).expect("tri").len(), 3);
    }

    #[test]
    fn triangulate_with_hole() {
        let outer = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        let hole = vec![
            Point2::new(3.0, 3.0),
            Point2::new(7.0, 3.0),
            Point2::new(7.0, 7.0),
            Point2::new(3.0, 7.0),
        ];
        let idx = triangulate_polygon_with_holes(&outer, &[hole]).expect("tri");
        assert!(idx.len() > 6);
        assert_eq!(idx.len() % 3, 0);
    }

    #[test]
    fn polygon_normal_xy_plane() {
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let n = calculate_polygon_normal(&pts);
        assert!((n.z.abs() - 1.0).abs() < 0.001);
    }
}
