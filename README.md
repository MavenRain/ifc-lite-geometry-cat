# ifc-lite-geometry-cat

Geometry processing and mesh generation for IFC models, built on
[comp-cat-rs](https://github.com/MavenRain/comp-cat-rs).

## Overview

This crate transforms IFC geometry representations into GPU-ready
triangle meshes.  All effectful operations are expressed through the
`comp-cat-rs` effect system (`Io`), keeping geometry logic pure and
composable.

**Capabilities:**

- **Profile Handling** -- 2D profiles (rectangle, circle, hollow circle, arbitrary polygon)
- **Extrusion** -- 3D meshes from extruded profiles, returned as `Io<Error, Mesh>`
- **Triangulation** -- polygon triangulation with hole support via `earcutr`
- **Mesh Processing** -- merging, bounds, centroids, coordinate shifts
- **Transforms** -- 4x4 matrix application with correct normal transformation

## Quick Start

```rust
use ifc_lite_geometry_cat::{Profile2D, extrude_profile};

let profile = Profile2D::rectangle(2.0, 1.0);
let mesh = extrude_profile(&profile, 3.0, None).run().unwrap();
println!("{} vertices, {} triangles", mesh.vertex_count(), mesh.triangle_count());
```

## Functional Design

- **No mutation** -- all mesh operations return new values
- **`Io`-wrapped** -- extrusion returns `Io<Error, Mesh>` for composition via `map`/`flat_map`
- **Delay `run`** -- stay inside effects; call `run` only at the boundary
- **Pure triangulation** -- `triangulate_polygon` is a plain `&[Point2] -> Result<Vec<usize>>`

## Architecture

| Module          | Purpose                                          |
|-----------------|--------------------------------------------------|
| `mesh`          | `Mesh`, `CoordinateShift` -- GPU-ready data      |
| `profile`       | `Profile2D`, `ProfileType`, `Triangulation`      |
| `extrusion`     | `extrude_profile` -> `Io<Error, Mesh>`           |
| `triangulation` | Polygon triangulation, 3D projection, normals    |
| `error`         | Hand-rolled `Error` enum                         |

## License

Licensed under either of

- Apache License, Version 2.0
- MIT license

at your option.
