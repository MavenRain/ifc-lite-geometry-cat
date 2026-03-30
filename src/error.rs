//! Hand-rolled error type for geometry operations.

/// Errors that can occur during geometry processing.
#[derive(Debug)]
pub enum Error {
    /// Polygon triangulation failed.
    Triangulation(String),
    /// A 2D profile was invalid (too few points, degenerate, etc.).
    InvalidProfile(String),
    /// Extrusion parameters were invalid (non-positive depth, etc.).
    InvalidExtrusion(String),
    /// An operation produced an empty mesh.
    EmptyMesh(String),
    /// A general geometry processing error.
    Geometry(String),
    /// An error propagated from the core parser.
    Core(ifc_lite_core_cat::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Triangulation(msg) => write!(f, "triangulation failed: {msg}"),
            Self::InvalidProfile(msg) => write!(f, "invalid profile: {msg}"),
            Self::InvalidExtrusion(msg) => write!(f, "invalid extrusion parameters: {msg}"),
            Self::EmptyMesh(msg) => write!(f, "empty mesh: {msg}"),
            Self::Geometry(msg) => write!(f, "geometry processing error: {msg}"),
            Self::Core(e) => write!(f, "core parser error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(e) => Some(e),
            Self::Triangulation(_)
            | Self::InvalidProfile(_)
            | Self::InvalidExtrusion(_)
            | Self::EmptyMesh(_)
            | Self::Geometry(_) => None,
        }
    }
}

impl From<ifc_lite_core_cat::Error> for Error {
    fn from(e: ifc_lite_core_cat::Error) -> Self {
        Self::Core(e)
    }
}

impl Error {
    /// Create a geometry processing error.
    #[must_use]
    pub fn geometry(msg: impl Into<String>) -> Self {
        Self::Geometry(msg.into())
    }
}
