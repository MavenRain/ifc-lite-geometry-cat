//! Extrusion: turn 2D profiles into 3D meshes.
//!
//! The main entry point is [`extrude_profile`] which returns an
//! [`Io<Error, Mesh>`](comp_cat_rs::effect::io::Io) so that extrusion
//! composes with other effects via `flat_map` and `run` is deferred
//! to the boundary.

use comp_cat_rs::effect::io::Io;
use nalgebra::{Matrix4, Point3, Vector3};

use crate::error::Error;
use crate::mesh::Mesh;
use crate::profile::{Profile2D, Triangulation};

/// Extrude a 2D profile along the Z axis by `depth`.
///
/// Produces bottom and top caps plus side walls.  An optional 4x4
/// transform is applied to the final mesh.
///
/// # Errors
///
/// Returns [`Error::InvalidExtrusion`] when `depth <= 0` and
/// [`Error::Triangulation`] if the profile cannot be triangulated.
///
/// # Examples
///
/// ```
/// use ifc_lite_geometry_cat::{Profile2D, extrude_profile};
///
/// let rect = Profile2D::rectangle(2.0, 1.0);
/// let mesh = extrude_profile(&rect, 3.0, None).run().unwrap();
/// assert!(mesh.triangle_count() > 0);
/// ```
#[must_use]
pub fn extrude_profile(
    profile: &Profile2D,
    depth: f64,
    transform: Option<Matrix4<f64>>,
) -> Io<Error, Mesh> {
    let profile = profile.clone();
    Io::suspend(move || {
        (depth > 0.0)
            .then_some(())
            .ok_or_else(|| Error::InvalidExtrusion("depth must be positive".into()))?;

        let tri = profile.triangulate()?;
        let bottom = create_cap(&tri, 0.0, Vector3::new(0.0, 0.0, -1.0), true);
        let top = create_cap(&tri, depth, Vector3::new(0.0, 0.0, 1.0), false);
        let sides = create_sides(profile.outer(), depth);
        let hole_sides = Mesh::merge_all(
            &profile
                .holes()
                .iter()
                .map(|h| create_sides(h, depth))
                .collect::<Vec<_>>(),
        );

        let combined = Mesh::merge_all(&[bottom, top, sides, hole_sides]);

        Ok(transform.map(|mat| apply_transform(&combined, &mat)).unwrap_or(combined))
    })
}

// ── cap construction ────────────────────────────────────────────────

#[allow(clippy::cast_possible_truncation)]
fn create_cap(tri: &Triangulation, z: f64, normal: Vector3<f64>, flip: bool) -> Mesh {
    let z_f32 = z as f32;
    let nx = normal.x as f32;
    let ny = normal.y as f32;
    let nz = normal.z as f32;

    let positions: Vec<f32> = tri
        .points()
        .iter()
        .flat_map(|p| [p.x as f32, p.y as f32, z_f32])
        .collect();

    let normals: Vec<f32> = (0..tri.points().len())
        .flat_map(|_| [nx, ny, nz])
        .collect();

    let indices: Vec<u32> = tri
        .indices()
        .chunks(3)
        .flat_map(|chunk| if flip {
            [chunk[0] as u32, chunk[2] as u32, chunk[1] as u32]
        } else {
            [chunk[0] as u32, chunk[1] as u32, chunk[2] as u32]
        })
        .collect();

    Mesh::new(positions, normals, indices)
}

// ── side wall construction ──────────────────────────────────────────

#[allow(clippy::cast_possible_truncation)]
fn create_sides(boundary: &[nalgebra::Point2<f64>], depth: f64) -> Mesh {
    let n = boundary.len();
    if n < 2 {
        Mesh::empty()
    } else {
        let edges: Vec<_> = (0..n)
            .filter_map(|i| {
                let j = (i + 1) % n;
                let p0 = &boundary[i];
                let p1 = &boundary[j];
                let edge = Vector3::new(p1.x - p0.x, p1.y - p0.y, 0.0);
                let normal = Vector3::new(-edge.y, edge.x, 0.0).try_normalize(1e-10)?;
                Some((p0, p1, normal))
            })
            .collect();

        let depth_f32 = depth as f32;

        let positions: Vec<f32> = edges
            .iter()
            .flat_map(|(p0, p1, _)| {
                [
                    p0.x as f32, p0.y as f32, 0.0_f32,
                    p1.x as f32, p1.y as f32, 0.0_f32,
                    p1.x as f32, p1.y as f32, depth_f32,
                    p0.x as f32, p0.y as f32, depth_f32,
                ]
            })
            .collect();

        let normals: Vec<f32> = edges
            .iter()
            .flat_map(|(_, _, n)| {
                let nx = n.x as f32;
                let ny = n.y as f32;
                let nz = n.z as f32;
                [nx, ny, nz, nx, ny, nz, nx, ny, nz, nx, ny, nz]
            })
            .collect();

        let indices: Vec<u32> = (0..edges.len())
            .flat_map(|q| {
                let base = u32::try_from(q * 4).unwrap_or(0);
                [base, base + 1, base + 2, base, base + 2, base + 3]
            })
            .collect();

        Mesh::new(positions, normals, indices)
    }
}

// ── transform ───────────────────────────────────────────────────────

/// Apply a 4x4 transformation matrix, returning a new mesh.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn apply_transform(mesh: &Mesh, transform: &Matrix4<f64>) -> Mesh {
    let normal_matrix = transform
        .try_inverse()
        .unwrap_or(*transform)
        .transpose();

    let positions: Vec<f32> = mesh
        .positions()
        .chunks_exact(3)
        .flat_map(|c| {
            let p = transform.transform_point(&Point3::new(
                f64::from(c[0]),
                f64::from(c[1]),
                f64::from(c[2]),
            ));
            [p.x as f32, p.y as f32, p.z as f32]
        })
        .collect();

    let normals: Vec<f32> = mesh
        .normals()
        .chunks_exact(3)
        .flat_map(|c| {
            let n = Vector3::new(f64::from(c[0]), f64::from(c[1]), f64::from(c[2]));
            let t = (normal_matrix * n.to_homogeneous()).xyz().normalize();
            [t.x as f32, t.y as f32, t.z as f32]
        })
        .collect();

    Mesh::new(positions, normals, mesh.indices().to_vec())
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrude_rectangle() {
        let p = Profile2D::rectangle(10.0, 5.0);
        let mesh = extrude_profile(&p, 20.0, None).run().expect("extrude");
        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);

        let (lo, hi) = mesh.bounds();
        assert!((lo.x - -5.0).abs() < 0.01);
        assert!((hi.x - 5.0).abs() < 0.01);
        assert!((lo.z - 0.0).abs() < 0.01);
        assert!((hi.z - 20.0).abs() < 0.01);
    }

    #[test]
    fn extrude_with_translation() {
        let p = Profile2D::rectangle(10.0, 5.0);
        let t = Matrix4::new_translation(&Vector3::new(100.0, 200.0, 300.0));
        let mesh = extrude_profile(&p, 20.0, Some(t)).run().expect("extrude");

        let (lo, hi) = mesh.bounds();
        assert!((lo.x - 95.0).abs() < 0.01);
        assert!((hi.x - 105.0).abs() < 0.01);
        assert!((lo.z - 300.0).abs() < 0.01);
        assert!((hi.z - 320.0).abs() < 0.01);
    }

    #[test]
    fn extrude_circle() {
        let p = Profile2D::circle(5.0);
        let mesh = extrude_profile(&p, 10.0, None).run().expect("extrude");
        assert!(mesh.vertex_count() > 0);

        let (lo, hi) = mesh.bounds();
        assert!((lo.x - -5.0).abs() < 0.1);
        assert!((hi.x - 5.0).abs() < 0.1);
    }

    #[test]
    fn extrude_hollow_circle() {
        let p = Profile2D::hollow_circle(10.0, 5.0);
        let mesh = extrude_profile(&p, 15.0, None).run().expect("extrude");
        assert!(mesh.triangle_count() > 20);
    }

    #[test]
    fn negative_depth_errors() {
        let p = Profile2D::rectangle(10.0, 5.0);
        let result = extrude_profile(&p, -1.0, None).run();
        assert!(result.is_err());
    }
}
