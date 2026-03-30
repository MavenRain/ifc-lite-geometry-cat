//! Triangle mesh data structures.
//!
//! [`Mesh`] is the primary output of geometry processing.  It stores
//! GPU-ready vertex positions, normals, and triangle indices.  All
//! construction is purely functional; no `&mut self` methods exist.

use nalgebra::{Point3, Vector3};

// ═══════════════════════════════════════════════════════════════════
// CoordinateShift
// ═══════════════════════════════════════════════════════════════════

/// Offset subtracted from world coordinates before `f32` conversion
/// to preserve sub-millimetre precision for large (georeferenced)
/// coordinates.
#[derive(Debug, Clone, Copy)]
pub struct CoordinateShift {
    x: f64,
    y: f64,
    z: f64,
}

impl CoordinateShift {
    /// Create a shift from explicit offsets.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Create from a [`Point3`].
    #[must_use]
    pub fn from_point(p: Point3<f64>) -> Self {
        Self {
            x: p.x,
            y: p.y,
            z: p.z,
        }
    }

    /// `true` when the offset exceeds 10 km on any axis.
    #[must_use]
    pub fn is_significant(&self) -> bool {
        const THRESHOLD: f64 = 10_000.0;
        self.x.abs() > THRESHOLD || self.y.abs() > THRESHOLD || self.z.abs() > THRESHOLD
    }

    /// `true` when all components are exactly zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0
    }

    /// X offset.
    #[must_use]
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Y offset.
    #[must_use]
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Z offset.
    #[must_use]
    pub fn z(&self) -> f64 {
        self.z
    }
}

impl Default for CoordinateShift {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Mesh
// ═══════════════════════════════════════════════════════════════════

/// A triangle mesh with positions, normals, and indices.
///
/// All fields are private.  Construct via the associated functions
/// ([`Mesh::new`], [`Mesh::empty`], [`Mesh::from_triangle`]) and
/// combine via [`Mesh::merge`] or [`Mesh::merge_all`].
///
/// # Examples
///
/// ```
/// use ifc_lite_geometry_cat::Mesh;
///
/// let a = Mesh::new(
///     vec![0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0],
///     vec![0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0],
///     vec![0, 1, 2],
/// );
/// assert_eq!(a.vertex_count(), 3);
/// assert_eq!(a.triangle_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct Mesh {
    positions: Vec<f32>,
    normals: Vec<f32>,
    indices: Vec<u32>,
}

impl Mesh {
    /// Construct from raw buffers.
    #[must_use]
    pub fn new(positions: Vec<f32>, normals: Vec<f32>, indices: Vec<u32>) -> Self {
        Self {
            positions,
            normals,
            indices,
        }
    }

    /// An empty mesh.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Build a single-triangle mesh.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_triangle(
        v0: &Point3<f64>,
        v1: &Point3<f64>,
        v2: &Point3<f64>,
        normal: &Vector3<f64>,
    ) -> Self {
        let positions = vec![
            v0.x as f32, v0.y as f32, v0.z as f32,
            v1.x as f32, v1.y as f32, v1.z as f32,
            v2.x as f32, v2.y as f32, v2.z as f32,
        ];
        let n = vec![
            normal.x as f32, normal.y as f32, normal.z as f32,
            normal.x as f32, normal.y as f32, normal.z as f32,
            normal.x as f32, normal.y as f32, normal.z as f32,
        ];
        Self::new(positions, n, vec![0, 1, 2])
    }

    // ── Accessors ───────────────────────────────────────────────────

    /// Raw vertex positions (x, y, z interleaved).
    #[must_use]
    pub fn positions(&self) -> &[f32] {
        &self.positions
    }

    /// Raw vertex normals (nx, ny, nz interleaved).
    #[must_use]
    pub fn normals(&self) -> &[f32] {
        &self.normals
    }

    /// Triangle indices (i0, i1, i2 interleaved).
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Number of vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// Number of triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// `true` when the mesh contains no geometry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    // ── Pure combinators ────────────────────────────────────────────

    /// Merge two meshes into a new one.  Index offsets are adjusted
    /// automatically.
    #[must_use]
    pub fn merge(a: &Self, b: &Self) -> Self {
        match (a.is_empty(), b.is_empty()) {
            (true, true) => Self::empty(),
            (true, false) => b.clone(),
            (false, true) => a.clone(),
            (false, false) => {
                let offset = u32::try_from(a.positions.len() / 3).unwrap_or(0);
                let positions = a
                    .positions
                    .iter()
                    .chain(b.positions.iter())
                    .copied()
                    .collect();
                let normals = a
                    .normals
                    .iter()
                    .chain(b.normals.iter())
                    .copied()
                    .collect();
                let indices = a
                    .indices
                    .iter()
                    .copied()
                    .chain(b.indices.iter().map(|&i| i + offset))
                    .collect();
                Self::new(positions, normals, indices)
            }
        }
    }

    /// Merge a slice of meshes into one.
    #[must_use]
    pub fn merge_all(meshes: &[Self]) -> Self {
        meshes
            .iter()
            .filter(|m| !m.is_empty())
            .fold(Self::empty(), |acc, m| Self::merge(&acc, m))
    }

    /// Axis-aligned bounding box: `(min, max)`.
    #[must_use]
    pub fn bounds(&self) -> (Point3<f32>, Point3<f32>) {
        self.positions
            .chunks_exact(3)
            .fold(
                (
                    Point3::new(f32::MAX, f32::MAX, f32::MAX),
                    Point3::new(f32::MIN, f32::MIN, f32::MIN),
                ),
                |(lo, hi), chunk| {
                    (
                        Point3::new(lo.x.min(chunk[0]), lo.y.min(chunk[1]), lo.z.min(chunk[2])),
                        Point3::new(hi.x.max(chunk[0]), hi.y.max(chunk[1]), hi.z.max(chunk[2])),
                    )
                },
            )
    }

    /// Centroid in `f64` precision (useful for RTC offset calculation).
    #[must_use]
    pub fn centroid_f64(&self) -> Point3<f64> {
        let count = self.vertex_count();
        if count == 0 {
            Point3::origin()
        } else {
                let sum = self.positions.chunks_exact(3).fold(
                    Point3::new(0.0_f64, 0.0_f64, 0.0_f64),
                    |acc, chunk| {
                        Point3::new(
                            acc.x + f64::from(chunk[0]),
                            acc.y + f64::from(chunk[1]),
                            acc.z + f64::from(chunk[2]),
                        )
                    },
                );
                #[allow(clippy::cast_precision_loss)]
                let n = count as f64;
                Point3::new(sum.x / n, sum.y / n, sum.z / n)
        }
    }

    /// Apply a [`CoordinateShift`], returning a new mesh.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn with_shift(&self, shift: &CoordinateShift) -> Self {
        if shift.is_zero() {
            self.clone()
        } else {
                let positions = self
                    .positions
                    .chunks_exact(3)
                    .flat_map(|chunk| {
                        [
                            (f64::from(chunk[0]) - shift.x) as f32,
                            (f64::from(chunk[1]) - shift.y) as f32,
                            (f64::from(chunk[2]) - shift.z) as f32,
                        ]
                    })
                    .collect();
            Self::new(positions, self.normals.clone(), self.indices.clone())
        }
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::empty()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mesh() {
        let m = Mesh::empty();
        assert!(m.is_empty());
        assert_eq!(m.vertex_count(), 0);
        assert_eq!(m.triangle_count(), 0);
    }

    #[test]
    fn from_triangle_creates_one_tri() {
        let m = Mesh::from_triangle(
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(0.0, 1.0, 0.0),
            &Vector3::z(),
        );
        assert_eq!(m.vertex_count(), 3);
        assert_eq!(m.triangle_count(), 1);
    }

    #[test]
    fn merge_two_meshes() {
        let a = Mesh::new(
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0],
        );
        let b = Mesh::new(
            vec![1.0, 1.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0],
        );
        let merged = Mesh::merge(&a, &b);
        assert_eq!(merged.vertex_count(), 2);
        assert_eq!(merged.indices(), &[0, 1]);
    }

    #[test]
    fn centroid_computation() {
        let m = Mesh::new(
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            vec![0.0; 9],
            vec![0, 1, 2],
        );
        let c = m.centroid_f64();
        assert!((c.x - 10.0).abs() < 0.001);
        assert!((c.y - 10.0).abs() < 0.001);
    }

    #[test]
    fn shift_precision() {
        let m = Mesh::new(
            vec![500_000.0_f32, 5_000_000.0, 0.0, 500_010.0, 5_000_010.0, 10.0],
            vec![0.0; 6],
            vec![0, 1, 0],
        );
        let shift = CoordinateShift::new(500_000.0, 5_000_000.0, 0.0);
        let shifted = m.with_shift(&shift);
        assert!((shifted.positions()[0]).abs() < 0.001);
        assert!((shifted.positions()[3] - 10.0).abs() < 0.001);
    }
}
